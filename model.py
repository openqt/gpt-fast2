# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    flex_attention,
)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None
    # DeepSeek MLA params
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    # DeepSeek MoE params
    n_routed_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    routed_scaling_factor: float = 1.0
    # shared
    n_embd_head_kv: Optional[int] = None
    n_expert: Optional[int] = None
    n_shared_expert: Optional[int] = None
    n_activated: Optional[int] = None
    moe: Optional[dict] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head
        # Convert moe dict to individual attributes if present
        if self.moe is not None:
            m = self.moe
            self.n_routed_experts = m.get('n_routed_experts')
            self.n_activated_experts = m.get('n_activated_experts')
            self.n_shared_experts = m.get('n_shared_experts')
            self.moe_intermediate_size = m.get('moe_intermediate_size')
            self.n_group = m.get('n_group')
            self.topk_group = m.get('topk_group')
            self.routed_scaling_factor = m.get('routed_scaling_factor', 1.0)

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
            
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "Mistral-7B": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),

    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-70b": dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.1-405b": dict(block_size=131072, n_layer=126, n_head=128, n_local_heads=8, dim=16384, intermediate_size=53248, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    "llama-3.2-1b": dict(block_size=131072, n_layer=16, n_head=32, n_local_heads=8, dim=2048, intermediate_size=8192, vocab_size=128256, rope_base=500000,
        rope_scaling=dict(factor=32.0, low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192),
    ),
    # DeepSeek-V2
    "DeepSeek-V2": dict(
        block_size=4096, vocab_size=102400, n_layer=60,
        dim=5120, n_head=128, head_dim=128,
        kv_lora_rank=512, qk_nope_head_dim=64, qk_rope_head_dim=64,
        v_head_dim=128, intermediate_size=12288,
        n_routed_experts=160, n_activated_experts=6, n_shared_experts=2,
        n_group=8, topk_group=4,
        moe_intermediate_size=1536,
        norm_eps=1e-6, rope_base=10000,
        routed_scaling_factor=16.0,
    ),
    # DeepSeek-V3
    "DeepSeek-V3": dict(
        block_size=8192, vocab_size=129280, n_layer=61,
        dim=7168, n_head=128, head_dim=128,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, intermediate_size=18432,
        n_routed_experts=256, n_activated_experts=8, n_shared_experts=1,
        n_group=8, topk_group=4,
        moe_intermediate_size=2048,
        norm_eps=1e-6, rope_base=10000,
        routed_scaling_factor=16.0,
    ),
    # DeepSeek-R1
    "DeepSeek-R1": dict(
        block_size=8192, vocab_size=129280, n_layer=61,
        dim=7168, n_head=128, head_dim=128,
        kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
        v_head_dim=128, intermediate_size=18432,
        n_routed_experts=256, n_activated_experts=8, n_shared_experts=1,
        n_group=8, topk_group=4,
        moe_intermediate_size=2048,
        norm_eps=1e-6, rope_base=10000,
        routed_scaling_factor=16.0,
    ),
}


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        print(f">>> cache_shape (max_batch_size, n_heads, max_seq_length, head_dim): {cache_shape}")
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class MLAKVCache(nn.Module):
    """KV cache for Multi-head Latent Attention.
    
    Stores compressed KV latent (c) and RoPE part of K (k_rope),
    significantly reducing cache size compared to standard MHA/GQA.
    """
    def __init__(self, max_batch_size, max_seq_length, kv_lora_rank,
                 n_head, qk_rope_head_dim, v_head_dim, dtype=torch.bfloat16):
        super().__init__()
        c_cache = torch.zeros(max_batch_size, max_seq_length, kv_lora_rank, dtype=dtype)
        self.register_buffer('c_cache', c_cache)
        k_rope_cache = torch.zeros(max_batch_size, n_head, max_seq_length, qk_rope_head_dim, dtype=dtype)
        self.register_buffer('k_rope_cache', k_rope_cache)

    def update(self, input_pos, c_val, k_rope_val):
        # c_val: [B, S, D_c], k_rope_val: [B, H, S, D_rope]
        assert input_pos.shape[0] == c_val.shape[1]
        self.c_cache[:, input_pos] = c_val
        self.k_rope_cache[:, :, input_pos] = k_rope_val
        return self.c_cache, self.k_rope_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        weight_dtype = self.output.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            weight_dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            weight_dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, weight_dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, weight_dtype, self.config.rope_scaling)

    def forward(self, mask: BlockMask, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        config = ModelArgs.from_name(name)
        # Route to DeepSeek model if MLA params are present
        if config.kv_lora_rank is not None:
            return DeepSeekModel(config)
        # Route to Mixtral model if expert params are present
        if config.n_expert is not None or config.num_experts is not None:
            from mixtral_moe.model import Transformer as MixtralModel
            return MixtralModel(config)
        return cls(config)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: BlockMask) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache : Optional[KVCache] = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: BlockMask, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = flex_attention(q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads))

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class MLA(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2/V3).
    
    Uses low-rank KV compression via learned down/up projections.
    The KV cache stores only the compressed latent (c) and RoPE part of K,
    dramatically reducing cache size vs standard MHA/GQA.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_head = config.n_head
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.dim = config.dim

        # Query projections (nope + rope parts)
        self.wq_nope = nn.Linear(config.dim, config.n_head * config.qk_nope_head_dim, bias=False)
        self.wq_rope = nn.Linear(config.dim, config.n_head * config.qk_rope_head_dim, bias=False)

        # KV compression (joint down-projection)
        self.wkv = nn.Linear(config.dim, config.kv_lora_rank, bias=False)

        # K up-projections (from compressed latent)
        self.wk_nope = nn.Linear(config.kv_lora_rank, config.n_head * config.qk_nope_head_dim, bias=False)
        self.wk_rope = nn.Linear(config.kv_lora_rank, config.n_head * config.qk_rope_head_dim, bias=False)

        # V up-projection (from compressed latent)
        self.wv = nn.Linear(config.kv_lora_rank, config.n_head * config.v_head_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(config.n_head * config.v_head_dim, config.dim, bias=False)

        self.kv_cache: Optional[MLAKVCache] = None

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: BlockMask, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        # Compressed KV latent [B, S, kv_lora_rank]
        c = self.wkv(x)

        # K projections from compressed latent
        k_nope = self.wk_nope(c)  # [B, S, n_head * qk_nope_head_dim]
        k_rope = self.wk_rope(c)  # [B, S, n_head * qk_rope_head_dim]

        # V projection from compressed latent
        v = self.wv(c)  # [B, S, n_head * v_head_dim]

        # Q projections
        q_nope = self.wq_nope(x)  # [B, S, n_head * qk_nope_head_dim]
        q_rope = self.wq_rope(x)  # [B, S, n_head * qk_rope_head_dim]

        # Reshape to [B, S, H, D]
        q_nope = q_nope.view(bsz, seqlen, self.n_head, self.qk_nope_head_dim)
        q_rope = q_rope.view(bsz, seqlen, self.n_head, self.qk_rope_head_dim)
        k_nope = k_nope.view(bsz, seqlen, self.n_head, self.qk_nope_head_dim)
        k_rope = k_rope.view(bsz, seqlen, self.n_head, self.qk_rope_head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.v_head_dim)

        # Apply RoPE to rope parts only (nope parts are position-agnostic)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        k_rope = apply_rotary_emb(k_rope, freqs_cis)

        # Concatenate nope + rope
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, S, n_head, qk_head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, S, n_head, qk_head_dim]

        q = q.transpose(1, 2)  # [B, n_head, S, qk_head_dim]
        k = k.transpose(1, 2)  # [B, n_head, S, qk_head_dim]
        v = v.transpose(1, 2)  # [B, n_head, S, v_head_dim]

        if self.kv_cache is not None:
            # Update cache with compressed latent (c) and rope part of K
            c_cache, k_rope_cache = self.kv_cache.update(input_pos, c, k_rope.transpose(1, 2))
            # Reconstruct full K and V from cached compressed latent
            c_full = c_cache.view(bsz * self.kv_cache.c_cache.shape[1], -1)
            k_nope_full = self.wk_nope(c_full).view(bsz, -1, self.n_head, self.qk_nope_head_dim).transpose(1, 2)
            v_full = self.wv(c_full).view(bsz, -1, self.n_head, self.v_head_dim).transpose(1, 2)
            k = torch.cat([k_nope_full, k_rope_cache], dim=-1)
            v = v_full

        y = flex_attention(q, k, v, block_mask=mask)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        y = self.wo(y)
        return y


class DeepSeekBlock(nn.Module):
    """Transformer block with MLA + DeepSeekMoE."""
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attention = MLA(config)
        self.block_sparse_moe = DeepSeekMoE(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: BlockMask) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.block_sparse_moe(self.ffn_norm(h))
        return out


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE with fine-grained experts + shared experts.
    
    Uses routed experts with group-limited top-k routing and shared experts.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.moe_intermediate_size = config.moe_intermediate_size
        self.routed_scaling_factor = config.routed_scaling_factor

        # Shared experts (always active)
        self.shared_experts = FeedForward(config)
        if config.n_shared_experts > 1:
            # If multiple shared experts, stack them
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(config.n_shared_experts)
            ])

        # Routed experts (sparsely activated)
        self.gate = nn.Linear(config.dim, config.n_routed_experts, bias=False)
        self.experts = nn.Parameter(
            torch.empty(config.n_routed_experts, 3 * config.moe_intermediate_size, config.dim)
        )
        # Use separate w2 for each expert
        self.experts_w2 = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, config.moe_intermediate_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        x_flat = x.view(-1, self.dim)  # [T, D]

        # Shared experts
        if isinstance(self.shared_experts, nn.ModuleList):
            shared_out = sum(e(x) for e in self.shared_experts)
        else:
            shared_out = self.shared_experts(x)

        # Gate scores for routing
        scores = self.gate(x_flat)  # [T, E]
        scores_soft = F.softmax(scores.float(), dim=-1).type_as(scores)

        # Group-limited top-k routing
        if self.n_group and self.topk_group:
            experts_per_group = self.n_routed_experts // self.n_group  # E_g
            group_scores = scores_soft.view(
                x_flat.shape[0], self.n_group, experts_per_group
            ).max(dim=-1).values  # [T, n_group]
            _, topk_groups = torch.topk(group_scores, self.topk_group, dim=-1)  # [T, topk_group]

            # Create mask for selected groups
            group_mask = torch.zeros(
                x_flat.shape[0], self.n_group, device=scores.device, dtype=torch.bool
            )
            group_mask.scatter_(1, topk_groups, True)
            group_mask = group_mask[:, :, None].expand(-1, -1, experts_per_group).reshape(
                x_flat.shape[0], -1
            )
            scores_soft = scores_soft.masked_fill(~group_mask, float('-inf'))

        # Top-k expert selection within eligible groups
        expert_weights, expert_indices = torch.topk(
            scores_soft, self.n_activated_experts, dim=-1
        )  # [T, A], [T, A]
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Gather expert parameters
        T, A = expert_indices.shape
        w1 = self.experts[expert_indices]  # [T, A, 3*D_ff, D]
        w2 = self.experts_w2[expert_indices]  # [T, A, D, D_ff]

        # Split merged w1 into w1, w3 (swiGLU variant: gate and up projection)
        d_ff = self.moe_intermediate_size
        w1_gate = w1[:, :, :d_ff, :]  # [T, A, D_ff, D]
        w1_up = w1[:, :, d_ff:2*d_ff, :]  # [T, A, D_ff, D]

        # Compute expert outputs
        x_gate = torch.einsum('ti,taoi->tao', x_flat, w1_gate)
        x_up = torch.einsum('ti,taoi->tao', x_flat, w1_up)
        x_act = F.silu(x_gate) * x_up  # [T, A, D_ff]
        expert_out = torch.einsum('tao,taio->tai', x_act, w2)  # [T, A, D]

        # Weighted sum of expert outputs
        routed_out = torch.einsum('tai,ta->ti', expert_out, expert_weights)
        routed_out = routed_out * self.routed_scaling_factor

        return (routed_out.view(bsz, seqlen, -1) + shared_out)


class DeepSeekModel(nn.Module):
    """DeepSeek-V2/V3 model with MLA and DeepSeekMoE.
    
    Supports DeepSeek-V2, DeepSeek-V3, and DeepSeek-R1 architectures.
    Uses flex_attention for efficient GQA-style attention.
    """
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(DeepSeekBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        weight_dtype = self.output.weight.dtype
        if hasattr(self.output, "scales"):
            weight_dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            weight_dtype = self.output.scales_and_zeros.dtype

        for b in self.layers:
            b.attention.kv_cache = MLAKVCache(
                max_batch_size, max_seq_length,
                self.config.kv_lora_rank,
                self.config.n_head,
                self.config.qk_rope_head_dim,
                self.config.v_head_dim,
                weight_dtype,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.qk_rope_head_dim,  # RoPE only applied to rope part
            self.config.rope_base,
            weight_dtype,
            self.config.rope_scaling,
        )

    def forward(self, mask: BlockMask, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    # 词嵌入两两分组后，计算每个分组对应的旋转角度 theta
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    # 计算每个位置对应的旋转角度 m*theta
    m = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(m, freqs)
    # 计算每个位置对应的旋转角度对应的复数
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    # 将每个位置对应的旋转角度对应的复数，freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)

# 应用旋转角度
def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # [B, S, H, D/2, 2]
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
