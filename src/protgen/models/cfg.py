# Classifier-free guidance on top of the group's LSTM

import torch
from torch import nn

from ..data import FAMILIES
from .rnn import BOS, EOS, PAD, RNNEncoder, decode

# Family ids: AcyP=0, HPr=1, HU=2
#   id 3: NULL_FAMILY is the null condition used for unconditional
NULL_FAMILY = len(FAMILIES)

# Similar to RNNEncoder, adding family embedding
class ConditionalRNNEncoder(RNNEncoder):
    def __init__(self, *args, num_classes=len(FAMILIES), **kwargs):
        super().__init__(*args, **kwargs)

        self.config["num_classes"] = num_classes

        self.family_embed = nn.Embedding(
            num_classes + 1, self.embed.embedding_dim
        )
        nn.init.zeros_(self.family_embed.weight)

    def forward(self, x, state=None, y=None):
        if y is None:       # handling missing labels
            y = torch.full(
                (x.size(0),), NULL_FAMILY,
                dtype=torch.long, device=x.device
            )

        x = self.embed(x) + self.family_embed(y)[:, None, :]
        x = self.embed_drop(x)

        h_all, _ = self.lstm(x, state)
        h_all = self.norm(h_all)

        logits = self.l_out(self.out_drop(h_all))
        return logits, h_all

    def step(self, token, state=None, y=None):
        if y is None:
            y = torch.full(
                (token.size(0),), NULL_FAMILY,
                dtype=torch.long, device=token.device
            )

        x = self.embed(token) + self.family_embed(y)
        x = self.embed_drop(x).unsqueeze(1)

        h_all, new_state = self.lstm(x, state)
        h = self.norm(h_all[:, 0])

        logits = self.l_out(self.out_drop(h))
        return logits, h, new_state

# reusing the trained RNN
def from_unconditional(path, device="cpu", num_classes=len(FAMILIES)):
    ckpt = torch.load(path, map_location=device)
    model = ConditionalRNNEncoder(**ckpt["config"], num_classes=num_classes)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if unexpected or set(missing) != {"family_embed.weight"}:
        raise ValueError(f"not an rnn.RNNEncoder checkpoint: missing {missing}, unexpected {unexpected}")
    
    return model.to(device)

# loading a CFG checkpoint
def load_cfg(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = ConditionalRNNEncoder(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)

# training classifier-free guidance
def drop_condition(y, p):
    drop = torch.rand(y.shape, device=y.device) < p
    return torch.where(drop, torch.full_like(y, NULL_FAMILY), y)

@torch.no_grad()
def sample_cfg(model, n=100, family_id=0, guidance_scale=1.0, max_length=129, temperature=1.0):

    if family_id not in range(len(FAMILIES)):
        raise ValueError(f"family_id must be one of {list(range(len(FAMILIES)))}")
    
    model.eval()
    device = next(model.parameters()).device
    y = torch.cat([torch.full((n,), family_id, dtype=torch.long, device=device),
                   torch.full((n,), NULL_FAMILY, dtype=torch.long, device=device)])
    
    token = torch.full((n,), BOS, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    state, out = None, []

    # logits = logits_uncond + w * (logits_cond - logits_uncond)
    #     w=0: unconditional, w>=1: conditional + strengthening guidance
    for _ in range(max_length - 2):                
        logits, _, state = model.step(token.repeat(2), state, y)
        cond, uncond = logits[:n], logits[n:]
        guided = uncond + guidance_scale * (cond - uncond)
        token = torch.multinomial((guided / temperature).softmax(-1), 1).squeeze(1)
        token = token.masked_fill(done, PAD)            
        done |= token == EOS
        out.append(token)
        if done.all():
            break

    return [decode(seq) for seq in torch.stack(out, dim=1)]
