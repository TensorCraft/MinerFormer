import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import config

model_name = config.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(config.device)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (100000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # precompute sin/cos for max_seq_len positions
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i , j -> i j', t, inv_freq)  # [seq, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)             # duplicate for sin & cos
        self.register_buffer('sin', emb.sin()[None, None, :, :])  # [1,1,seq,head_dim]
        self.register_buffer('cos', emb.cos()[None, None, :, :])  # same shape

    def forward(self, x):
        # x: [batch, n_heads, seq_len, head_dim]
        seq_len = x.shape[2]
        return (x * self.cos[:, :, :seq_len, :]) + (rotate_half(x) * self.sin[:, :, :seq_len, :])

def rotate_half(x):
    # x: [..., head_dim]
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class AddNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, sublayer_out):
        u = x + sublayer_out
        return self.norm(u)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.d_model = embed_dim // num_heads
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_o = nn.Linear(embed_dim, embed_dim)
        self.rotary_emb = RotaryEmbedding(self.d_model, max_seq_len)
        self.add_norm = AddNorm(embed_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.proj_q(x).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1, 2)
        K = self.proj_k(x).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1, 2)
        V = self.proj_v(x).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1, 2)
        
        Q = self.rotary_emb(Q)
        K = self.rotary_emb(K)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_model ** 0.5
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        output = self.proj_o(output)
        return self.add_norm(x, output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_units, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_units),
            nn.Linear(ff_units, embed_dim),
            nn.Dropout(dropout)
        )
        self.add_norm = AddNorm(embed_dim)
    def forward(self, x):
        xt = self.ff(x)
        return self.add_norm(x, xt)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, max_seq_len, num_heads, ff_units, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.ff_units = ff_units
        self.dropout = dropout
        self.attention = MultiHeadAttention(embed_dim, num_heads, max_seq_len)
        self.ff = FeedForwardNetwork(embed_dim, ff_units=ff_units, dropout=dropout)
    def forward(self, x):
        x = self.attention(x)
        x = self.ff(x)
        return x

def normalize_price(x, idx, alpha=0.2, eps=1e-8):
    val = x[:, :, idx]
    min_val = val.min(dim=1, keepdim=True)[0]
    max_val = val.max(dim=1, keepdim=True)[0]
    range_val = max_val - min_val + eps
    norm_val = (val - min_val + eps) / (range_val * (1 + 2 * alpha) + eps)
    return norm_val, min_val, range_val

def denormalize_price(norm_val, min_val, range_val, alpha=0.2):
    return norm_val * (range_val * (1 + 2 * alpha)) + min_val - range_val * alpha

def normalize_simple(x, idx, eps=1e-8):
    val = x[:, :, idx]
    min_val = val.min(dim=1, keepdim=True)[0]
    max_val = val.max(dim=1, keepdim=True)[0]
    range_val = max_val - min_val + eps
    return (val - min_val) / range_val


class MinerFormer(nn.Module):
    def __init__(self, intervals, open_idx, close_idx, highest_idx, lowest_idx, dimmensions, llm_emebed_dim, max_seq_len, embed_dim, num_heads, ff_dim, num_layers, dropout):
        super(MinerFormer, self).__init__()
        self.llm_embed_dim = llm_emebed_dim
        self.intervals = intervals
        self.open_idx = open_idx
        self.close_idx = close_idx
        self.highest_idx = highest_idx
        self.lowest_idx = lowest_idx
        self.dimmensions = dimmensions
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.embeddings = nn.ModuleList([
            nn.Linear(dimmensions, embed_dim) for _ in range(intervals)
        ])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim*intervals+llm_emebed_dim, max_seq_len, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.ModuleList([
            nn.Linear(embed_dim, 2) for _ in range(intervals)
        ])

    def forward(self, inputs, llm_embedding):
        B, T = llm_embedding.shape[0], llm_embedding.shape[1]
        bar_embeddings = []
        outputs = []
        norm_info = []

        for i, x in enumerate(inputs):
            x = x.clone()
            D = x.shape[-1]
            bar_norm_info = []

            # Normalize each bar's features
            for idx in range(D):
                if idx in [self.open_idx, self.close_idx, self.highest_idx, self.lowest_idx]:
                    norm, min_val, range_val = normalize_price(x, idx)
                    x[:, :, idx] = norm
                    bar_norm_info.append((min_val, range_val))  # Special normalization for price features
                else:
                    norm = normalize_simple(x, idx)
                    x[:, :, idx] = norm
                    bar_norm_info.append(None)  # general normalization for other features

            norm_info.append(bar_norm_info)
            emb = self.embeddings[i](x)
            bar_embeddings.append(emb)

        combined = torch.cat(bar_embeddings + [llm_embedding], dim=-1)

        out = combined
        for block in self.transformer_blocks:
            out = block(out)

        for i in range(self.intervals):
            bar_out = self.output_layer[i](bar_embeddings[i])  # (B, T, 2)

            high_pred = bar_out[:, :, 0]
            low_pred = bar_out[:, :, 1]

            min_val_high, range_high = norm_info[i][self.highest_idx]
            min_val_low, range_low = norm_info[i][self.lowest_idx]

            real_high = denormalize_price(high_pred, min_val_high, range_high)
            real_low = denormalize_price(low_pred, min_val_low, range_low)

            outputs.append(torch.stack([real_high, real_low], dim=-1))  # (B, T, 2)

        return outputs  # List[(B, T, 2)]
