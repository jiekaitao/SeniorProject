"""
LoRA (Low-Rank Adaptation) for Tiny Recursive Models
Inspired by QLoRA: Efficient Finetuning of Quantized LLMs

LoRA decomposes weight matrices into low-rank matrices:
  W = W_base + BA
  where W_base is frozen, and BA is trainable

This significantly reduces:
1. Trainable parameters (80-99% reduction)
2. Memory usage
3. Training time
4. Gradient computation

Implementation for TRM components:
- Attention QKV projections
- Attention output projections  
- MLP gate/up/down projections
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from models.common import trunc_normal_init_


class LoRALinear(nn.Module):
    """
    LoRA-adapted linear layer
    Decomposes W into W_base (frozen) + BA (trainable)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank (typically 16-64)
        alpha: Scaling factor (typically equal to rank)
        dropout: LoRA dropout
    """
    def __init__(
        self,
        base_module: nn.Module,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.05,
        bias: bool = False,
    ):
        super().__init__()
        
        self.in_features = base_module.weight.shape[1]
        self.out_features = base_module.weight.shape[0]
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
        # Freeze base weights
        self.base_weight = base_module.weight.detach()
        if hasattr(base_module, 'bias') and base_module.bias is not None:
            self.base_bias = base_module.bias.detach()
        else:
            self.base_bias = None
            
        # LoRA matrices: A and B
        # A: [in_features, rank], initialized with Kaiming
        self.lora_A = nn.Parameter(
            torch.empty(self.in_features, rank)
        )
        # B: [rank, out_features], initialized to zero (so initial output is just W_base)
        self.lora_B = nn.Parameter(
            torch.zeros(rank, self.out_features)
        )
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base projection
        output = F.linear(x, self.base_weight, self.base_bias)
        
        # LoRA projection: x @ A @ B
        lora_output = self.lora_dropout(x) @ self.lora_A.to(x.dtype)
        lora_output = lora_output @ self.lora_B.to(x.dtype)
        
        # Scale and add
        output = output + self.scale * lora_output
        
        return output
    
    @property
    def trainable_params(self) -> int:
        """Count of trainable parameters"""
        return self.lora_A.numel() + self.lora_B.numel()
    
    @property
    def total_params(self) -> int:
        """Total parameters (base + LoRA)"""
        base_params = self.base_weight.numel() + (self.base_bias.numel() if self.base_bias is not None else 0)
        return base_params + self.trainable_params


class LoRAAttention(nn.Module):
    """
    LoRA-adapted attention module
    Applies LoRA to QKV projection
    """
    def __init__(
        self,
        attention_module: nn.Module,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        
        # Store original config
        self.hidden_size = attention_module.hidden_size
        self.head_dim = attention_module.head_dim
        self.output_size = attention_module.output_size
        self.num_heads = attention_module.num_heads
        self.num_key_value_heads = attention_module.num_key_value_heads
        self.causal = attention_module.causal
        
        # Replace QKV projection with LoRA
        self.qkv_proj_lo = LoRALinear(
            attention_module.qkv_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Replace output projection with LoRA
        self.o_proj_lo = LoRALinear(
            attention_module.o_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
    def forward(self, cos_sin: Tuple[torch.Tensor, torch.Tensor], hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection with LoRA
        qkv = self.qkv_proj_lo(hidden_states)
        
        # Split heads (same as original)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        
        # RoPE (same as original)
        if cos_sin is not None:
            cos, sin = cos_sin
            import einops
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        
        # Scaled dot-product attention
        query, key, value = map(
            lambda t: einops.rearrange(t, 'B S H D -> B H S D'),
            (query, key, value)
        )
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=self.causal)
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)
        
        # Output projection with LoRA
        return self.o_proj_lo(attn_output)


class LoRASwiGLU(nn.Module):
    """
    LoRA-adapted SwiGLU module
    Applies LoRA to gate_up and down projections
    """
    def __init__(
        self,
        swiglu_module: nn.Module,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        
        # Replace gate_up projection with LoRA
        self.gate_up_proj_lo = LoRALinear(
            swiglu_module.gate_up_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Replace down projection with LoRA
        self.down_proj_lo = LoRALinear(
            swiglu_module.down_proj,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
    def forward(self, x):
        # Gate/up projection with LoRA
        gate, up = self.gate_up_proj_lo(x).chunk(2, dim=-1)
        
        # Down projection with LoRA
        return self.down_proj_lo(F.silu(gate) * up)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary positional embeddings"""
    import einops
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def convert_block_to_lora(
    block: nn.Module,
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.05,
):
    """
    Convert a TinyRecursiveReasoningModel_ACTV1Block to use LoRA
    
    Args:
        block: The block to convert
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        
    Returns:
        Converted block
    """
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Block
    from models.layers import Attention, SwiGLU
    
    config = block.config
    
    # Convert attention to LoRA
    if hasattr(block, 'self_attn'):
        block.self_attn = LoRAAttention(
            block.self_attn,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    # Convert MLP to LoRA
    if hasattr(block, 'mlp'):
        block.mlp = LoRASwiGLU(
            block.mlp,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
    
    return block


def estimate_memory_reduction(
    num_blocks: int,
    hidden_size: int,
    expansion: float,
    rank: int = 32,
):
    """
    Estimate memory reduction from LoRA
    
    For a single block:
    - Attention QKV: hidden_size × (3 * hidden_size) params
    - Attention O: hidden_size × hidden_size params
    - MLP gate_up: hidden_size × (2 * inter_size) params
    - MLP down: inter_size × hidden_size params
    
    With LoRA rank=r:
    - Attention QKV LoRA: hidden_size × r + r × (3 * hidden_size)
    - Attention O LoRA: hidden_size × r + r × hidden_size
    - MLP gate_up LoRA: hidden_size × r + r × (2 * inter_size)
    - MLP down LoRA: inter_size × r + r × hidden_size
    
    Returns:
        reduction_ratio: How much smaller trainable params are
        trainable_params: Number of trainable params
    """
    inter_size = int(expansion * hidden_size * 2 / 3)
    
    # Original params per block
    attn_qkv = hidden_size * (3 * hidden_size)
    attn_o = hidden_size * hidden_size
    mlp_up = hidden_size * (2 * inter_size)
    mlp_down = inter_size * hidden_size
    
    original_per_block = attn_qkv + attn_o + mlp_up + mlp_down
    
    # LoRA params per block (counting A and B matrices)
    attn_qkv_lo = hidden_size * rank + rank * (3 * hidden_size)
    attn_o_lo = hidden_size * rank + rank * hidden_size
    mlp_up_lo = hidden_size * rank + rank * (2 * inter_size)
    mlp_down_lo = inter_size * rank + rank * hidden_size
    
    lora_per_block = attn_qkv_lo + attn_o_lo + mlp_up_lo + mlp_down_lo
    
    original_total = original_per_block * num_blocks
    lora_total = lora_per_block * num_blocks
    
    reduction_ratio = 1 - (lora_total / original_total)
    
    return reduction_ratio, lora_total


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '/data/TRM')
    
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Block, TinyRecursiveReasoningModel_ACTV1Config
    from pydantic import BaseModel
    
    # Create a test config
    config_dict = {
        'batch_size': 1,
        'seq_len': 100,
        'num_puzzle_identifiers': 1000,
        'vocab_size': 50,
        'H_cycles': 3,
        'L_cycles': 6,
        'H_layers': 0,
        'L_layers': 6,
        'hidden_size': 512,
        'expansion': 2.66,
        'num_heads': 16,
        'pos_encodings': 'rope',
        'halt_max_steps': 16,
        'halt_exploration_prob': 0.1,
        'puzzle_emb_ndim': 0
    }
    
    config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
    
    # Create a test block
    block = TinyRecursiveReasoningModel_ACTV1Block(config)
    
    # Convert to LoRA
    lora_block = convert_block_to_lora(block, rank=32, alpha=32.0)
    
    # Estimate reduction
    reduction, params = estimate_memory_reduction(
        num_blocks=6,
        hidden_size=512,
        expansion=2.66,
        rank=32
    )
    
    print(f"Memory reduction: {reduction * 100:.1f}%")
    print(f"Trainable params: {params:,}")
    print("✅ LoRA implementation ready!")



