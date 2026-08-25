"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekEngram(nn.Module):
    def __init__(self, table_size=65537, d_model=384, vocab_size=32768,
                 block_fraction=0.15, soft_scale_kept=True, capacity_factor=1.4,
                 capacity_ema_decay=0.98, idf_path="idf_weights.pt"):
        super().__init__()
        self.table_size = table_size
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Fraction of the VOCAB (by frequency rank, most-common-first) that is hard-blocked
        # from lookup-table access. Change at runtime with set_block_fraction().
        self.block_fraction = block_fraction
        # If True, kept (unblocked) tokens aren't all given full weight -- they're scaled
        # continuously by rarity, ramping from ~0 right above the block cutoff up to 1.0
        # for the single rarest token. If False, every kept token gets full weight (hard
        # binary keep/drop, no gradation). Toggle any time with set_soft_scale().
        self.soft_scale_kept = soft_scale_kept
        # Headroom multiplier applied to the *observed, running-average* number of kept
        # (unblocked) tokens per batch, used to size the static compute buffer.
        # NOTE: this is calibrated from real data (via an EMA below), not from
        # (1 - block_fraction) alone -- token frequency is Zipfian, so the true fraction
        # of *occurrences* that survive blocking is usually much smaller than the fraction
        # of *vocab* you block. See _get_capacity() for details.
        self.capacity_factor = capacity_factor
        self.capacity_ema_decay = capacity_ema_decay
        # capacity_ema[N] -> running average of observed num_keep for a given B*T size N.
        self._capacity_ema = {}
        # Diagnostics: how often we had to drop tokens because more tokens were "kept"
        # in a given step than the current capacity allows. If this creeps up, raise
        # capacity_factor.
        self.dropped_token_events = 0
        self.dropped_tokens_total = 0
        self.total_forward_calls = 0

        self.unigram_embd = nn.Embedding(vocab_size, d_model)
        self.bigram_embd = nn.Embedding(table_size, d_model)
        self.trigram_embd = nn.Embedding(table_size, d_model)

        self.mem_proj = Linear(d_model, d_model, bias=False)
        self.gate_proj = Linear(d_model, d_model)

        self.p1 = 131
        self.p2 = 13331

        # Continuous IDF signal (lower = more common). Used only to DERIVE the hard block mask
        # below (via percentile), not used as a soft multiplicative gate anymore.
        self.register_buffer("token_idf", torch.ones(vocab_size), persistent=False)
        # Hard binary mask: True = one of the `block_fraction` most common tokens, fully
        # skipped (no embedding lookup, no mem_proj/gate_proj compute) for that token.
        self.register_buffer("token_blocked", torch.zeros(vocab_size, dtype=torch.bool), persistent=False)
        # Continuous per-token multiplier applied to KEPT tokens' engram output (ignored,
        # left at 0, for blocked tokens since those are never gathered). 1.0 = the single
        # rarest token in the vocab; ramps down toward 0 just above the block cutoff.
        # Only used when soft_scale_kept=True. See _recompute_block_mask().
        self.register_buffer("token_scale", torch.ones(vocab_size), persistent=False)

    @torch.no_grad()
    def load_idf(self, idf_path="idf_weights.pt"):
        if self.token_idf.device.type == "meta":
            return

        if os.path.exists(idf_path):
            idf_weights = torch.load(idf_path, map_location="cpu")
            target_len = self.token_idf.shape[0]
            if idf_weights.shape[0] < target_len:
                padding = torch.zeros(target_len - idf_weights.shape[0])
                idf_weights = torch.cat([idf_weights, padding], dim=0)

            self.token_idf.copy_(idf_weights[:target_len].to(self.token_idf.device))
            print(f"[Engram] Loaded TF-IDF weights from '{idf_path}'")
        else:
            print(f"[Engram] Warning: '{idf_path}' not found. Initializing with uniform weights (1.0) — no tokens will be blocked.")
            self.token_idf.fill_(1.0)

        self._recompute_block_mask()

    @torch.no_grad()
    def _recompute_block_mask(self):
        """(Re)derive the hard token_blocked mask AND the continuous token_scale ramp
        from token_idf, block_fraction, and soft_scale_kept.

        Blocking uses a RANK-based (percentile) cutoff, not a value-based threshold:
        exactly `round(block_fraction * vocab_size)` tokens are blocked, namely those
        with the lowest IDF (= most frequent). This is what makes block_fraction mean
        what it says, regardless of how IDF happens to be distributed after normalization.

        Among the tokens that survive blocking, if soft_scale_kept is True, token_scale
        ramps linearly from just above 0 (right at the cutoff) to 1.0 (the single rarest
        token), so "kept" isn't all-or-nothing -- moderately common survivors still
        contribute less engram signal than genuinely rare ones. If soft_scale_kept is
        False, every kept token gets a flat scale of 1.0 (pure hard keep/drop).
        """
        idf = self.token_idf
        eps = 1e-8
        if idf.numel() == 0:
            return
        if idf.std() < 1e-6:
            print(f"[Engram] Warning: token_idf looks uninitialized/uniform — skipping hard blocking and soft scaling (0% blocked, scale=1.0 everywhere). Call load_idf() first.")
            self.token_blocked.zero_()
            self.token_scale.fill_(1.0)
            return

        n = idf.numel()
        # Allow k=0 (block_fraction<=0 => nothing blocked) but never block the entire vocab.
        k = max(0, min(int(round(self.block_fraction * n)), n - 1))

        mask = torch.zeros_like(idf, dtype=torch.bool)
        if k > 0:
            # topk on -idf == the k tokens with the SMALLEST idf (i.e. most common)
            _, blocked_idx = torch.topk(-idf, k)
            mask[blocked_idx] = True
            threshold = idf[blocked_idx].max()
        else:
            threshold = idf.min() - eps

        if self.soft_scale_kept:
            denom = max((idf.max() - threshold).item(), eps)
            scale = torch.clamp((idf - threshold) / denom, min=0.0, max=1.0)
        else:
            scale = torch.ones_like(idf)
        scale = scale.masked_fill(mask, 0.0)  # irrelevant for blocked tokens, zeroed for clarity

        self.token_blocked.copy_(mask)
        self.token_scale.copy_(scale)
        # Any change to the mask invalidates our capacity estimates (kept-token rate shifts).
        self._capacity_ema = {}
        self.dropped_token_events = 0
        self.dropped_tokens_total = 0
        mode = "soft-scaled" if self.soft_scale_kept else "flat (hard keep/drop)"
        print(f"[Engram] Hard-blocked {k}/{n} tokens ({100*self.block_fraction:.1f}% by frequency rank) from lookup-table access. "
              f"Remaining {n-k} tokens are {mode}.")

    def set_block_fraction(self, frac):
        """Change what fraction of the vocab (by frequency rank) is blocked, at runtime."""
        assert 0.0 <= frac < 1.0, "block_fraction must be in [0, 1)"
        self.block_fraction = frac
        self._recompute_block_mask()

    def set_soft_scale(self, enabled):
        """Toggle continuous rarity-scaling among kept tokens on/off at runtime."""
        self.soft_scale_kept = bool(enabled)
        self._recompute_block_mask()

    def compute_stats(self):
        """Human-readable summary of current calibration: how much compute is actually
        being saved, and whether capacity_factor needs to be raised (drops > 0)."""
        lines = [f"[Engram] block_fraction={self.block_fraction:.3f}  capacity_factor={self.capacity_factor}"]
        for N, ema in self._capacity_ema.items():
            cap = max(1, min(N, int(math.ceil(ema * self.capacity_factor))))
            lines.append(f"  shape N={N}: ema_kept={ema:.1f}  capacity={cap}  reduction={100*(1-cap/N):.1f}%")
        lines.append(f"  dropped_token_events={self.dropped_token_events}  dropped_tokens_total={self.dropped_tokens_total}")
        if self.dropped_tokens_total > 0:
            lines.append("  (drops > 0: consider raising capacity_factor)")
        return "\n".join(lines)

    def _get_capacity(self, N, num_keep):
        """Static compute-buffer size for this call, self-calibrated from a running
        average of the ACTUAL number of kept (unblocked) tokens seen for this B*T=N
        shape, rather than a theoretical (1 - block_fraction) estimate.

        Why not just use (1 - block_fraction) * N directly? Token frequency is Zipfian:
        blocking the top `block_fraction` of the VOCAB (by rank) typically removes the
        vast majority of token OCCURRENCES in real text (e.g. blocking the most common
        15% of vocab can cover 80%+ of occurrences), so the true kept rate is usually
        much lower than (1 - block_fraction) suggests. Sizing capacity off the
        pessimistic formula gives ~0% compute savings in practice. Sizing it off a
        running average of what's actually observed gives real, correctly-calibrated
        savings after a short warmup.

        The very first call for a given N uses the exact observed count (free, no
        padding or drops). After that, capacity is `ceil(ema * capacity_factor)`,
        updated with an EMA after every call so it stays close to the true rate.
        """
        ema = self._capacity_ema.get(N)
        if ema is None:
            # First time seeing this shape: exact fit, then start the running average.
            capacity = max(1, min(N, num_keep))
            self._capacity_ema[N] = float(num_keep)
        else:
            capacity = max(1, min(N, int(math.ceil(ema * self.capacity_factor))))
            self._capacity_ema[N] = (
                self.capacity_ema_decay * ema + (1.0 - self.capacity_ema_decay) * num_keep
            )
        return capacity

    def forward(self, idx, h_layer, kv_cache=None):
        dtype = h_layer.dtype
        B, T = idx.shape
        device = idx.device
        d_model = self.d_model
        self.total_forward_calls += 1

        if kv_cache is not None:
            if not hasattr(kv_cache, 'engram_history'):
                kv_cache.engram_history = idx
            else:
                if T == 1:
                    kv_cache.engram_history = torch.cat([kv_cache.engram_history, idx], dim=1)
                else:
                    kv_cache.engram_history = idx
            full_idx = kv_cache.engram_history
        else:
            full_idx = idx

        idx_m1 = torch.cat([torch.zeros((B, 1), dtype=full_idx.dtype, device=device), full_idx[:, :-1]], dim=1)
        idx_m2 = torch.cat([torch.zeros((B, 2), dtype=full_idx.dtype, device=device), full_idx[:, :-2]], dim=1)

        current_idx = full_idx[:, -T:]
        current_m1 = idx_m1[:, -T:]
        current_m2 = idx_m2[:, -T:]

        # N-gram rolling hashes still computed for every position (cheap integer ops) --
        # they're needed as context inputs even for positions we go on to skip below.
        bigram_hash = (current_m1 * self.p1 + current_idx) % self.table_size
        trigram_hash = (current_m2 * self.p2 + bigram_hash) % self.table_size

        # ---- Hard percentile-based blocking + sparse compute ----
        blocked = self.token_blocked[current_idx]          # (B, T) bool
        keep_mask_flat = (~blocked).reshape(-1)             # (N,)
        N = B * T

        if not keep_mask_flat.any():
            # Every token in this step is one of the blocked common words: skip the module entirely.
            return h_layer

        keep_positions = torch.nonzero(keep_mask_flat, as_tuple=False).squeeze(-1)  # (num_keep,), dynamic
        num_keep = keep_positions.numel()

        capacity = self._get_capacity(N, num_keep)

        if num_keep > capacity:
            self.dropped_token_events += 1
            self.dropped_tokens_total += (num_keep - capacity)
            keep_positions = keep_positions[:capacity]
            num_keep = capacity

        pad_len = capacity - num_keep
        if pad_len > 0:
            pad_idx = keep_positions.new_zeros(pad_len)
            gather_positions = torch.cat([keep_positions, pad_idx], dim=0)  # (capacity,)
        else:
            gather_positions = keep_positions

        valid_mask = torch.zeros(capacity, dtype=torch.bool, device=device)
        valid_mask[:num_keep] = True

        flat_current = current_idx.reshape(-1)
        flat_bigram = bigram_hash.reshape(-1)
        flat_trigram = trigram_hash.reshape(-1)
        flat_h = h_layer.reshape(N, d_model)

        g_current = flat_current.index_select(0, gather_positions)   # (capacity,)
        g_bigram = flat_bigram.index_select(0, gather_positions)
        g_trigram = flat_trigram.index_select(0, gather_positions)
        g_h = flat_h.index_select(0, gather_positions)                # (capacity, d_model)
        g_scale = self.token_scale[g_current].to(dtype=dtype)          # (capacity,) rarity ramp among kept tokens

        e_1gram = self.unigram_embd(g_current)
        e_2gram = self.bigram_embd(g_bigram)
        e_3gram = self.trigram_embd(g_trigram)
        e_t = (e_1gram + e_2gram + e_3gram).to(dtype=dtype)

        memory_features = self.mem_proj(e_t)             # (capacity, d_model) -- only `capacity` rows, not N
        gate = torch.sigmoid(self.gate_proj(g_h))          # (capacity, d_model)

        contribution = gate * memory_features * g_scale.unsqueeze(-1)
        contribution = contribution * valid_mask.unsqueeze(-1).to(dtype=dtype)  # zero out padding slots

        out_flat = torch.zeros(N, d_model, dtype=dtype, device=device)
        out_flat.index_add_(0, gather_positions, contribution)

        return h_layer + out_flat.reshape(B, T, d_model)

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = norm(q), norm(k) # QK norm
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })

        self.engram = DeepSeekEngram(d_model=config.n_embd, vocab_size=padded_vocab_size, table_size=8191)
        self.engram_enabled = os.environ.get("NANOCHAT_ENGRAM_ENABLED", "1") == "1"
        
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.
        """
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Initialize Engram module
        if hasattr(self, "engram") and self.engram is not None:
            if hasattr(self.engram, "unigram_embd"):
                torch.nn.init.normal_(self.engram.unigram_embd.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(self.engram.bigram_embd.weight, mean=0.0, std=0.02)
                torch.nn.init.normal_(self.engram.trigram_embd.weight, mean=0.0, std=0.02)
            elif hasattr(self.engram, "embeddings"):
                torch.nn.init.normal_(self.engram.embeddings.weight, mean=0.0, std=0.02)

            torch.nn.init.xavier_uniform_(self.engram.mem_proj.weight)
            torch.nn.init.zeros_(self.engram.gate_proj.weight)
        
            # Initialize gate bias to negative value so gate starts near 0 (~0.11)
            if self.engram.gate_proj.bias is not None:
                torch.nn.init.constant_(self.engram.gate_proj.bias, -2.0)

            # Load IDF weights
            if hasattr(self.engram, "load_idf"):
                self.engram.load_idf("idf_weights.pt")

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + sum(p.numel() for p in self.engram.parameters())
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all standard parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]

        # Safely extract Engram parameters (split into embeddings and projections)
        engram_embed_params = []
        engram_proj_params = []
        if hasattr(self, "engram") and self.engram is not None:
            for name, param in self.engram.named_parameters():
                if "embd" in name or "embeddings" in name:
                    engram_embed_params.append(param)
                else:
                    engram_proj_params.append(param)

        # Verify parameter coverage
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) + 
            len(value_embeds_params) + len(resid_params) + len(x0_params) + 
            len(smear_params) + len(engram_embed_params) + len(engram_proj_params)
        )

        # Scale LR for AdamW parameters ∝ 1 / sqrt(d_model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]

        # Engram Embeddings (AdamW with ZERO weight decay)
        if engram_embed_params:
            param_groups.append(
                dict(kind='adamw', params=engram_embed_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0)
            )
    
        # Engram Projections (mem_proj, gate_proj)
        if engram_proj_params:
            param_groups.append(
                dict(kind='adamw', params=engram_proj_params, lr=matrix_lr * 0.5, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.01)
            )

        # Muon groups for transformer matrices
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        if T > self.cos.size(1):
            repeats = (T // self.cos.size(1)) + 1
            repeat_sizes = [1] * self.cos.dim()
            repeat_sizes[1] = repeats
            self.cos = self.cos.repeat(*repeat_sizes)
            self.sin = self.sin.repeat(*repeat_sizes)
            
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            # INJECT ENGRAM HERE:
            if i == 2 and hasattr(self, "engram") and self.engram is not None and getattr(self, "engram_enabled", True):
                x = self.engram(idx, x, kv_cache=kv_cache)
            
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
