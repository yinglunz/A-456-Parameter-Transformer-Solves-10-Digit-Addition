"""Low-rank transformer for 10-digit addition.

Builds on the gpt-acc-jax 777-param record architecture (d=7, 1 layer, 1 head,
d_ff=14, tied embeddings, no bias), adding low-rank factorizations to position
embeddings and optionally QKV/attention-output projections.
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_layer: int = 1
    d_model: int = 7
    n_head: int = 1
    d_ff: int = 14
    dropout: float = 0.0
    max_seq_len: int = 33   # 22 prompt + 11 target digits
    vocab_size: int = 14    # 0-9, +, =, <PAD>, <EOS>
    # Low-rank ranks (0 = full rank)
    pos_rank: int = 0
    qkv_rank: int = 0
    attn_out_rank: int = 0
    ffn_rank: int = 0       # 0 = full rank FFN


# ---------------------------------------------------------------------------
# Low-rank building blocks
# ---------------------------------------------------------------------------
class LowRankLinear(nn.Module):
    """y = x @ A @ B  (no bias). Params: in_f*r + r*out_f."""

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))
        # Init so that A@B has variance ~= 2/(in+out)
        nn.init.normal_(self.A, std=math.sqrt(2.0 / (in_features + rank)))
        nn.init.normal_(self.B, std=math.sqrt(2.0 / (rank + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


class LowRankEmbedding(nn.Module):
    """E[i] = A[i] @ B. Params: num_emb*r + r*emb_dim."""

    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.A = nn.Parameter(torch.empty(num_embeddings, rank))
        self.B = nn.Parameter(torch.empty(rank, embedding_dim))
        nn.init.normal_(self.A, std=0.02)
        nn.init.normal_(self.B, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.A) @ self.B


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int,
                 qkv_rank: int = 0, attn_out_rank: int = 0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        if qkv_rank > 0:
            self.qkv = LowRankLinear(d_model, 3 * d_model, qkv_rank)
        else:
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        if attn_out_rank > 0:
            self.proj = LowRankLinear(d_model, d_model, attn_out_rank)
        else:
            self.proj = nn.Linear(d_model, d_model, bias=False)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:seqlen, :seqlen], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(bsz, seqlen, d)
        return self.proj(y)


# ---------------------------------------------------------------------------
# Feed-forward (no bias, matching gpt-acc-jax)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, ffn_rank: int = 0):
        super().__init__()
        if ffn_rank > 0:
            self.fc1 = LowRankLinear(d_model, d_ff, ffn_rank)
            self.fc2 = LowRankLinear(d_ff, d_model, ffn_rank)
        else:
            self.fc1 = nn.Linear(d_model, d_ff, bias=False)
            self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Transformer block (pre-norm, matching gpt-acc-jax)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_head, cfg.max_seq_len,
            qkv_rank=cfg.qkv_rank, attn_out_rank=cfg.attn_out_rank,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, ffn_rank=cfg.ffn_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class TinyDecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding (always full rank; only 14 * d_model params)
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Position embedding (optionally low-rank)
        if cfg.pos_rank > 0:
            self.pos_emb = LowRankEmbedding(cfg.max_seq_len, cfg.d_model, cfg.pos_rank)
        else:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight-tied output head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seqlen = idx.shape
        pos = torch.arange(seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        out = prompt
        for _ in range(max_new_tokens):
            idx = out[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(idx)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
        return out


def count_parameters(model: nn.Module) -> int:
    """Count unique parameters (respects weight tying)."""
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total
