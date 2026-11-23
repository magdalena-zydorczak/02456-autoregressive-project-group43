# Classifier-free guidance on top of the group's LSTM

import torch
from torch import nn

from ..data import FAMILIES
from .rnn import RNNEncoder, sample

# Family ids: AcyP=0, HPr=1, HU=2
#   id 3: NULL_FAMILY is the null condition used for unconditional
NULL_FAMILY = len(FAMILIES)

# Similar to RNNEncoder, adding family embedding
class ConditionalRNNEncoder(RNNEncoder):
    def __init__(self, *args, num_classes=len(FAMILIES), **kwargs):
        super().__init__(*args, **kwargs)
        self.config["num_classes"] = num_classes                            # to store hyperparameters

        self.family_embed = nn.Embedding(num_classes + 1, self.embed.embedding_dim)   # one row per family + null
        nn.init.zeros_(self.family_embed.weight)                            # start as the unconditional RNN

    # same as RNNEncoder.forward, with the family added to every token
    def forward(self, x, state=None, y=None):
        if y is None:                                                       # no family given -> unconditional
            y = torch.full((x.size(0),), NULL_FAMILY, dtype=torch.long, device=x.device)

        x = self.embed(x) + self.family_embed(y)[:, None, :]
        x = self.embed_drop(x)

        h_all, _ = self.lstm(x, state)
        h_all = self.norm(h_all)

        logits = self.l_out(self.out_drop(h_all))
        return logits, h_all

    # sampling per token
    def step(self, token, state=None, y=None):
        if y is None:
            y = torch.full((token.size(0),), NULL_FAMILY, dtype=torch.long, device=token.device)

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

# both conditions in one batch, so rnn.sample sees a plain model with guided logits
#   logits = logits_uncond + w * (logits_cond - logits_uncond)
#       w=0: unconditional, w>=1: conditional + strengthening guidance
class GuidedModel(nn.Module):
    def __init__(self, model, family_id, guidance_scale):
        super().__init__()
        self.model = model
        self.family_id = family_id
        self.guidance_scale = guidance_scale

    def step(self, token, state=None):
        n = token.size(0)
        device = token.device
        y = torch.cat([torch.full((n,), self.family_id, dtype=torch.long, device=device),
                       torch.full((n,), NULL_FAMILY, dtype=torch.long, device=device)])

        logits, h, new_state = self.model.step(token.repeat(2), state, y)   # one call for both branches
        cond, uncond = logits[:n], logits[n:]
        return uncond + self.guidance_scale * (cond - uncond), h[:n], new_state

# generate new protein sequences for one family
@torch.no_grad()
def sample_cfg(model, n=100, family_id=0, guidance_scale=1.0, max_length=129, temperature=1.0):
    if family_id not in range(len(FAMILIES)):
        raise ValueError(f"family_id must be one of {list(range(len(FAMILIES)))}")

    guided = GuidedModel(model, family_id, guidance_scale)                  # reuse the sampler from phase 1
    return sample(guided, n=n, max_length=max_length, temperature=temperature)
