import torch
from torch import nn

from ..data import AA, total_tokens

# special token IDs and lookup dict
PAD, BOS, EOS = 0, 1, 2
_idx_to_token = {i: t for i, t in enumerate(total_tokens)}

# build encoder = LSTM 
class RNNEncoder(nn.Module):
    def __init__(self, vocab_size=len(total_tokens), emb_dim=64, hidden=256, num_layers=2):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, emb_dim=emb_dim,          # to store hyperparamters
                           hidden=hidden, num_layers=num_layers)
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)     # turn tokenIDs into vectors
        self.embed_drop = nn.Dropout(0.1) # dropout on the embedding layer
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)     # store sequence info as hidden state
        self.norm = nn.LayerNorm(hidden)
        self.out_drop = nn.Dropout(0.2)
        self.l_out = nn.Linear(hidden, vocab_size, bias=False)              # get logits from hidden state
        
        # Weight initialization
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name: # input weights
                nn.init.kaiming_normal_(p, nonlinearity="tanh")
            elif "weight_hh" in name: # hidden weights
                for w in p.data.chunk(4, dim=0):    
                    nn.init.orthogonal_(w)
            elif "bias" in name: # biases
                nn.init.zeros_(p)
                if "bias_ih" in name:
                    p.data[hidden:2 * hidden] = 1.0 

    # train with seeing true previous tokens
    def forward(self, x, state=None):
        x = self.embed(x)                        
        x = self.embed_drop(x)

        h_all, _ = self.lstm(x, state)              
        h_all = self.norm(h_all)

        logits = self.l_out(self.out_drop(h_all))   
        return logits, h_all

    # sampling per token
    def step(self, token, state=None):
        x = self.embed(token).unsqueeze(1)          
        x = self.embed_drop(x)

        h_all, new_state = self.lstm(x, state)      
        h = self.norm(h_all[:, 0])                  

        logits = self.l_out(self.out_drop(h))      
        return logits, h, new_state

# get AAs from tokenIDs
def decode(tokens):
    residues = set(AA)
    return "".join(t for t in (_idx_to_token[int(i)] for i in tokens) if t in residues)


# generate new protein sequences
@torch.no_grad()
def sample(model, n=100, max_length=129, temperature=1.0, logit_fn=None):

    # setup
    model.eval()
    device = next(model.parameters()).device
    token = torch.full((n,), BOS, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    state, out = None, []

    # generate sequence token by token
    for i in range(max_length - 2):                     # no BOS and EOS tokens
        logits, _, state = model.step(token, state)     # get next token logits
        if logit_fn is not None:
            logits = logit_fn(logits, state, i)         # optional classifier guidance
        token = torch.multinomial((logits / temperature).softmax(-1), 1).squeeze(1)     # sample one token
        token = token.masked_fill(done, PAD)            # fill with PAD once EOS is reached
        done |= token == EOS
        out.append(token)
        if done.all():
            break

    return [decode(seq) for seq in torch.stack(out, dim=1)]
