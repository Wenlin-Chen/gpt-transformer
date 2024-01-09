import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class MHSA(nn.Module):

    def __init__(self, n_heads, embed_size, block_size, dropout, bias, causal):
        super().__init__()

        self.head_size = embed_size // n_heads
        
        self.kqv = nn.Linear(embed_size, 3*embed_size, bias=False)
        self.att_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_size, embed_size, bias=bias)
        self.proj_dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.causal = causal
        if causal:
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        # x: (B, T, E)
        B, T, E = x.shape # (B, T, E)

        k, q, v = self.kqv(x).split(E, dim=-1)  # each (B, T, E)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, H, T, E//H)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, H, T, E//H)
        v= v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, H, T, E//H)


        att_weights = q @ k.transpose(-2, -1) * self.head_size**(-0.5)  # (B, H, T, T)
        if self.causal:
            att_weights = att_weights.masked_fill(self.tril[:, :, :T, :T]==0, float('-inf'))  # (B, H, T, T)
        att_weights = F.softmax(att_weights, dim=-1)  # (B, H, T, T)
        att_weights = self.att_dropout(att_weights) # (B, H, T, T)

        x_att = att_weights @ v  # (B, H, T, E//H)
        x_att = x_att.transpose(1, 2).contiguous().view(B, T, E)  # (B, T, E)
        out = self.proj_dropout(self.proj(x_att))   # (B, T, E)

        return out
    

class FeedForward(nn.Module):

    def __init__(self, embed_size, dropout, bias):
        super().__init__()

        self.fc = nn.Linear(embed_size, 4*embed_size, bias=bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(4*embed_size, embed_size, bias=bias)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x
    

class TransformerBlock(nn.Module):

    def __init__(self, n_heads, embed_size, block_size, dropout, bias, causal):
        super().__init__()

        self.mhsa = MHSA(n_heads, embed_size, block_size, dropout, bias, causal)
        self.ffn = FeedForward(embed_size, dropout, bias)
        self.ln1 = nn.LayerNorm(embed_size, bias=bias)
        self.ln2 = nn.LayerNorm(embed_size, bias=bias)

    def forward(self, x):
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x
    
class GPT(nn.Module):

    def __init__(self, vocab_size, n_layers, n_heads, embed_size, block_size, dropout, bias):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embed = nn.Embedding(vocab_size, embed_size)
        self.position_embed = nn.Embedding(block_size, embed_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(n_heads, embed_size, block_size, dropout, bias, causal=True) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_size, bias=bias)
        self.head = nn.Linear(embed_size, vocab_size, bias=False)

        self.token_embed.weight = self.head.weight

        self.apply(self._init_weights)
        for n, p in self.named_parameters():
            if n.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02*(2.0*n_layers)**(-0.5))

        print("Number of parameters: {}M".format(self.get_num_params()/1e6))

    def get_num_params(self, non_embedding=True):
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.position_embed.weight.numel()
        return num_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        B, T = x.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)

        tok_emb = self.token_embed(x)  # (B, T, E)
        pos_emb = self.position_embed(pos)  # (T, E)
        emb = tok_emb + pos_emb  # (B, T, E)
        x = self.emb_dropout(emb)  # (B, T, E)
        for block in self.transformer_blocks:
            x = block(x)  # (B, T, E)
        x = self.ln(x)  # (B, T, E)
        
        if y is None:
            logits = self.head(x[:, [-1], :])  # (B, 1, E)
            loss = None

        else:
            logits = self.head(x)  # (B, T, V)
            loss = F.cross_entropy(logits.view(B*T, -1), y.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0):
        for _ in tqdm(range(max_new_tokens)):
            B, T = x.shape
            if T > self.block_size:
                x_truncated = x[:, -self.block_size:]
            else:
                x_truncated = x
            logits, _ = self(x_truncated)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, x_next], dim=-1)

        return x

