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
    use_rmsnorm: bool = False   # True = RMSNorm (no bias), False = LayerNorm
    tie_qkv: str = "none"      # "none","all","qk","kv","shareA","shareB","shareB_tieQK","shareB_tieKV","shareA_tieKV","shareA_tieQK"


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


class RMSNorm(nn.Module):
    """Root Mean Square normalization (weight only, no bias).

    Saves d_model params per instance vs LayerNorm by removing bias.
    """

    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, max_seq_len: int,
                 qkv_rank: int = 0, attn_out_rank: int = 0,
                 tie_qkv: str = "none"):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.tie_qkv = tie_qkv

        if tie_qkv == "all":
            # Single shared projection: Q=K=V=Wx
            if qkv_rank > 0:
                self.qkv_shared = LowRankLinear(d_model, d_model, qkv_rank)
            else:
                self.qkv_shared = nn.Linear(d_model, d_model, bias=False)
        elif tie_qkv == "qk":
            # Shared Q=K, separate V
            if qkv_rank > 0:
                self.qk_shared = LowRankLinear(d_model, d_model, qkv_rank)
                self.v_proj = LowRankLinear(d_model, d_model, qkv_rank)
            else:
                self.qk_shared = nn.Linear(d_model, d_model, bias=False)
                self.v_proj = nn.Linear(d_model, d_model, bias=False)
        elif tie_qkv == "kv":
            # Separate Q, shared K=V
            if qkv_rank > 0:
                self.q_proj = LowRankLinear(d_model, d_model, qkv_rank)
                self.kv_shared = LowRankLinear(d_model, d_model, qkv_rank)
            else:
                self.q_proj = nn.Linear(d_model, d_model, bias=False)
                self.kv_shared = nn.Linear(d_model, d_model, bias=False)
        elif tie_qkv == "shareA":
            # Q,K,V share matrix A but have separate B matrices
            # x @ A_shared @ B_q, x @ A_shared @ B_k, x @ A_shared @ B_v
            assert qkv_rank > 0, "shareA requires qkv_rank > 0"
            self.qkv_A = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Bq = nn.Parameter(torch.empty(qkv_rank, d_model))
            self.qkv_Bk = nn.Parameter(torch.empty(qkv_rank, d_model))
            self.qkv_Bv = nn.Parameter(torch.empty(qkv_rank, d_model))
            std_a = math.sqrt(2.0 / (d_model + qkv_rank))
            std_b = math.sqrt(2.0 / (qkv_rank + d_model))
            nn.init.normal_(self.qkv_A, std=std_a)
            nn.init.normal_(self.qkv_Bq, std=std_b)
            nn.init.normal_(self.qkv_Bk, std=std_b)
            nn.init.normal_(self.qkv_Bv, std=std_b)
        elif tie_qkv == "shareB":
            # Q,K,V have separate A matrices but share B
            assert qkv_rank > 0, "shareB requires qkv_rank > 0"
            self.qkv_Aq = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Ak = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Av = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_B = nn.Parameter(torch.empty(qkv_rank, d_model))
            std_a = math.sqrt(2.0 / (d_model + qkv_rank))
            std_b = math.sqrt(2.0 / (qkv_rank + d_model))
            nn.init.normal_(self.qkv_Aq, std=std_a)
            nn.init.normal_(self.qkv_Ak, std=std_a)
            nn.init.normal_(self.qkv_Av, std=std_a)
            nn.init.normal_(self.qkv_B, std=std_b)
        elif tie_qkv == "shareB_tieQK":
            # Share B + tie A for Q=K: Aqk(shared), Av(separate), B(shared)
            assert qkv_rank > 0, "shareB_tieQK requires qkv_rank > 0"
            self.qkv_Aqk = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Av = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_B = nn.Parameter(torch.empty(qkv_rank, d_model))
            std_a = math.sqrt(2.0 / (d_model + qkv_rank))
            std_b = math.sqrt(2.0 / (qkv_rank + d_model))
            nn.init.normal_(self.qkv_Aqk, std=std_a)
            nn.init.normal_(self.qkv_Av, std=std_a)
            nn.init.normal_(self.qkv_B, std=std_b)
        elif tie_qkv == "shareB_tieKV":
            # Share B + tie A for K=V: Aq(separate), Akv(shared), B(shared)
            assert qkv_rank > 0, "shareB_tieKV requires qkv_rank > 0"
            self.qkv_Aq = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Akv = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_B = nn.Parameter(torch.empty(qkv_rank, d_model))
            std_a = math.sqrt(2.0 / (d_model + qkv_rank))
            std_b = math.sqrt(2.0 / (qkv_rank + d_model))
            nn.init.normal_(self.qkv_Aq, std=std_a)
            nn.init.normal_(self.qkv_Akv, std=std_a)
            nn.init.normal_(self.qkv_B, std=std_b)
        elif tie_qkv == "shareA_tieKV":
            # Shared A, tie B for K=V: h = x@A; Q = h@B_q; K = V = h@B_kv
            assert qkv_rank > 0, "shareA_tieKV requires qkv_rank > 0"
            self.qkv_A = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Bq = nn.Parameter(torch.empty(qkv_rank, d_model))
            self.qkv_Bkv = nn.Parameter(torch.empty(qkv_rank, d_model))
            std_a = math.sqrt(2.0 / (d_model + qkv_rank))
            std_b = math.sqrt(2.0 / (qkv_rank + d_model))
            nn.init.normal_(self.qkv_A, std=std_a)
            nn.init.normal_(self.qkv_Bq, std=std_b)
            nn.init.normal_(self.qkv_Bkv, std=std_b)
        elif tie_qkv == "shareA_tieQK":
            # Shared A, tie B for Q=K: h = x@A; Q = K = h@B_qk; V = h@B_v
            assert qkv_rank > 0, "shareA_tieQK requires qkv_rank > 0"
            self.qkv_A = nn.Parameter(torch.empty(d_model, qkv_rank))
            self.qkv_Bqk = nn.Parameter(torch.empty(qkv_rank, d_model))
            self.qkv_Bv = nn.Parameter(torch.empty(qkv_rank, d_model))
            std_a = math.sqrt(2.0 / (d_model + qkv_rank))
            std_b = math.sqrt(2.0 / (qkv_rank + d_model))
            nn.init.normal_(self.qkv_A, std=std_a)
            nn.init.normal_(self.qkv_Bqk, std=std_b)
            nn.init.normal_(self.qkv_Bv, std=std_b)
        else:
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

        if self.tie_qkv == "all":
            shared = self.qkv_shared(x)
            q = k = v = shared
        elif self.tie_qkv == "qk":
            qk = self.qk_shared(x)
            q = k = qk
            v = self.v_proj(x)
        elif self.tie_qkv == "kv":
            q = self.q_proj(x)
            kv = self.kv_shared(x)
            k = v = kv
        elif self.tie_qkv == "shareA":
            h = x @ self.qkv_A  # shared bottleneck
            q = h @ self.qkv_Bq
            k = h @ self.qkv_Bk
            v = h @ self.qkv_Bv
        elif self.tie_qkv == "shareB":
            q = (x @ self.qkv_Aq) @ self.qkv_B
            k = (x @ self.qkv_Ak) @ self.qkv_B
            v = (x @ self.qkv_Av) @ self.qkv_B
        elif self.tie_qkv == "shareB_tieQK":
            qk = (x @ self.qkv_Aqk) @ self.qkv_B
            q = k = qk
            v = (x @ self.qkv_Av) @ self.qkv_B
        elif self.tie_qkv == "shareB_tieKV":
            q = (x @ self.qkv_Aq) @ self.qkv_B
            kv = (x @ self.qkv_Akv) @ self.qkv_B
            k = v = kv
        elif self.tie_qkv == "shareA_tieKV":
            h = x @ self.qkv_A
            q = h @ self.qkv_Bq
            k = v = h @ self.qkv_Bkv
        elif self.tie_qkv == "shareA_tieQK":
            h = x @ self.qkv_A
            q = k = h @ self.qkv_Bqk
            v = h @ self.qkv_Bv
        else:
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
        norm_cls = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln1 = norm_cls(cfg.d_model)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_head, cfg.max_seq_len,
            qkv_rank=cfg.qkv_rank, attn_out_rank=cfg.attn_out_rank,
            tie_qkv=cfg.tie_qkv,
        )
        self.ln2 = norm_cls(cfg.d_model)
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
        norm_cls = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln_f = norm_cls(cfg.d_model)

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
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

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
