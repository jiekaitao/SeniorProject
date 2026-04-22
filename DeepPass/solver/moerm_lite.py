"""
MoERM-Lite: Mixture of External Reasoning Modules (Lite version)

Architecture (from GPT-5.4 Pro consultation):
  - N=4 experts (each a SolverCore)
  - Sequence-level soft routing (attention-pooled, not per-token)
  - Fixed K (no adaptive depth in v1)
  - Perceiver-style fixed-slot fusion into 32 output slots
  - Soft routing during training, top-2 at inference

Build order (GPT-5.4):
  1. Single solver scaling (12M/25M/42M) ← RUNNING NOW
  2. Shared-core expert-code baseline ← IMPLEMENTED BELOW
  3. MoERM-Lite ← IMPLEMENTED BELOW

Key math (from GPT-5.4):
  Router gradient is advantage-based:
    ∂ℓ/∂z_i = α_i(⟨g, M̄_i⟩ - Σ_j α_j⟨g, M̄_j⟩)
  Dead experts get no gradient (α_i ≈ 0).
  Collapse gives zero router gradient (all M̄_i equal).
  Need symmetry breaking: expert-specific init states.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base SolverCore
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from model import SolverCore, RMSNorm, BidirectionalBlock, CrossAttention


# ============================================================
# Component 1: Sequence-Level Router
# ============================================================
class SequenceRouter(nn.Module):
    """
    Sequence-level router: examines prompt embeddings and outputs
    soft gate weights over N experts.

    Uses attention pooling (learned query) rather than mean pooling
    for a richer summary of the input.
    """
    def __init__(self, d_input: int, n_experts: int, hidden: int = 512, n_heads: int = 8):
        super().__init__()
        self.n_experts = n_experts

        # Attention pooling: learned query attends over prompt embeddings
        self.query = nn.Parameter(torch.randn(1, 1, d_input) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_input, n_heads, batch_first=True)

        # Gate MLP: pooled representation → expert logits
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )

        # Temperature for training (anneal from 2.0 → 1.0)
        self.temperature = 2.0

    def forward(self, prompt_emb: torch.Tensor):
        """
        prompt_emb: (B, T, D) from frozen LLM embedding
        Returns: gate (B, N) soft routing weights, gate_logits (B, N)
        """
        B = prompt_emb.shape[0]
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(
            q.to(prompt_emb.dtype),
            prompt_emb,
            prompt_emb,
            need_weights=False
        )
        h = pooled[:, 0]  # (B, D)

        gate_logits = self.gate_mlp(h.to(self.query.dtype))  # (B, N)
        gate = F.softmax(gate_logits / self.temperature, dim=-1)

        return gate, gate_logits


# ============================================================
# Component 2: Slot Fusion
# ============================================================
class SlotFusion(nn.Module):
    """
    Perceiver-style fixed-slot cross-attention fusion.
    Takes memory outputs from multiple experts and compresses
    them into a fixed number of output slots.

    Each expert's memory is tagged with an expert-ID embedding
    and scaled by the gate weight.
    """
    def __init__(self, d_model: int, n_experts: int, n_out_slots: int = 32, n_heads: int = 8):
        super().__init__()
        self.n_out_slots = n_out_slots

        # Learned output slot queries
        self.slot_queries = nn.Parameter(
            torch.randn(1, n_out_slots, d_model) * 0.02
        )

        # Expert-ID embeddings (symmetry breaker for fusion)
        self.expert_id = nn.Embedding(n_experts, d_model)

        # Cross-attention: queries=slots, KV=expert memories
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Post-fusion FFN (use RMSNorm for dtype compatibility)
        self.norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, expert_mems: list, gate: torch.Tensor):
        """
        expert_mems: list of N tensors, each (B, M_i, D)
        gate: (B, N) routing weights
        Returns: fused memory (B, n_out_slots, D)
        """
        B = gate.size(0)
        kv_pieces = []

        for i, mem_i in enumerate(expert_mems):
            # Add expert-ID embedding
            mem_tagged = mem_i + self.expert_id.weight[i].unsqueeze(0).unsqueeze(0)
            # Scale by gate weight
            mem_scaled = gate[:, i].view(B, 1, 1) * mem_tagged
            kv_pieces.append(mem_scaled)

        kv = torch.cat(kv_pieces, dim=1)  # (B, N*M, D)
        q = self.slot_queries.expand(B, -1, -1).to(kv.dtype)

        # Cross-attention fusion
        fused, _ = self.cross_attn(q, kv, kv, need_weights=False)

        # Residual FFN
        fused = fused + self.ffn(self.norm(fused)).to(fused.dtype)

        return fused


# ============================================================
# Component 3: Shared-Core Expert-Code Baseline
# ============================================================
class SharedCoreExperts(nn.Module):
    """
    Simplest specialization test: ONE shared SolverCore with N
    expert-specific initial states (H_init, L_init_scale) and
    optional FiLM modulations.

    This is the cheapest real test of whether specialization helps.
    If this matches full MoERM, use this simpler model.
    """
    def __init__(self, n_experts: int = 4, d_model: int = 512, n_heads: int = 8,
                 ffn_dim: int = 1024, n_L_layers: int = 2, n_memory_slots: int = 32):
        super().__init__()
        self.n_experts = n_experts
        self.n_memory_slots = n_memory_slots

        # Shared solver core
        self.solver = SolverCore(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim,
                                 n_L_layers=n_L_layers, n_memory_slots=n_memory_slots)

        # Expert-specific initial states (symmetry breakers)
        self.expert_H_inits = nn.ParameterList([
            nn.Parameter(torch.randn(1, n_memory_slots, d_model) * 0.02)
            for _ in range(n_experts)
        ])
        self.expert_L_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1 + 0.05 * i))
            for i in range(n_experts)
        ])

    def forward(self, prompt_embeddings, expert_idx, K_inner=4, K_outer=3, grad_last_only=True):
        """
        Run shared solver with expert-specific initialization.
        expert_idx: integer, which expert's init to use
        """
        # Swap in expert-specific inits
        orig_H = self.solver.H_init.data.clone()
        orig_L = self.solver.L_init_scale.data.clone()

        self.solver.H_init.data = self.expert_H_inits[expert_idx].data
        self.solver.L_init_scale.data = self.expert_L_scales[expert_idx].data

        memory = self.solver(prompt_embeddings, K_inner=K_inner, K_outer=K_outer,
                            grad_last_only=grad_last_only)

        # Restore
        self.solver.H_init.data = orig_H
        self.solver.L_init_scale.data = orig_L

        return memory


# ============================================================
# Component 4: Full MoERM-Lite
# ============================================================
class MoERMLite(nn.Module):
    """
    Mixture of External Reasoning Modules — Lite version.

    N=4 experts, sequence-level soft routing, fixed K,
    Perceiver-style fusion into 32 output slots.

    Pipeline:
      prompt_emb → Router → select experts → run experts → fuse → memory tokens
    """
    def __init__(self, n_experts: int = 4, d_solver: int = 512, n_heads: int = 8,
                 ffn_dim: int = 1024, n_L_layers: int = 2,
                 n_memory_slots_per_expert: int = 32,
                 n_output_slots: int = 32,
                 llm_dim: int = 4096,
                 shared_core: bool = False):
        super().__init__()
        self.n_experts = n_experts
        self.n_output_slots = n_output_slots
        self.llm_dim = llm_dim
        self.shared_core = shared_core

        # Router
        self.router = SequenceRouter(
            d_input=llm_dim, n_experts=n_experts, hidden=512, n_heads=8
        )

        if shared_core:
            # Shared-core with expert codes
            self.experts = SharedCoreExperts(
                n_experts=n_experts, d_model=d_solver, n_heads=n_heads,
                ffn_dim=ffn_dim, n_L_layers=n_L_layers,
                n_memory_slots=n_memory_slots_per_expert
            )
        else:
            # Full separate experts
            self.experts = nn.ModuleList([
                SolverCore(d_model=d_solver, n_heads=n_heads, ffn_dim=ffn_dim,
                           n_L_layers=n_L_layers, n_memory_slots=n_memory_slots_per_expert)
                for _ in range(n_experts)
            ])

        # Fusion: compress N expert outputs into fixed output slots
        self.fusion = SlotFusion(
            d_model=llm_dim, n_experts=n_experts,
            n_out_slots=n_output_slots, n_heads=8
        )

        # Output normalization (match LLM embedding norm)
        self.out_norm = RMSNorm(llm_dim)

    def forward(self, prompt_embeddings, K_inner=4, K_outer=3,
                grad_last_only=True, top_k=None):
        """
        prompt_embeddings: (B, T, llm_dim) from frozen LLM embedding
        Returns: memory_tokens (B, n_output_slots, llm_dim)
        """
        B = prompt_embeddings.shape[0]

        # 1. Route
        gate, gate_logits = self.router(prompt_embeddings)  # (B, N)

        # 2. Run experts
        expert_mems = []
        if self.shared_core:
            for i in range(self.n_experts):
                mem_i = self.experts(prompt_embeddings, expert_idx=i,
                                    K_inner=K_inner, K_outer=K_outer,
                                    grad_last_only=grad_last_only)
                expert_mems.append(mem_i)
        else:
            for i, expert in enumerate(self.experts):
                # Optionally skip experts with very low gate weight (inference optimization)
                if top_k is not None and not self.training:
                    topk_vals, topk_idx = gate.topk(top_k, dim=-1)
                    if i not in topk_idx[0].tolist():
                        # Create zero memory for skipped experts
                        mem_i = torch.zeros(B, expert.n_memory_slots, self.llm_dim,
                                          device=prompt_embeddings.device, dtype=prompt_embeddings.dtype)
                        expert_mems.append(mem_i)
                        continue

                mem_i = expert(prompt_embeddings, K_inner=K_inner, K_outer=K_outer,
                              grad_last_only=grad_last_only)
                expert_mems.append(mem_i)

        # 3. Fuse
        memory = self.fusion(expert_mems, gate)

        # 4. Normalize
        memory = self.out_norm(memory)

        return memory, gate, gate_logits

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================
# Regularization losses (from GPT-5.4 consultation)
# ============================================================
def router_regularizers(gate, target_prior=None):
    """
    gate: (B, N) soft routing weights

    Returns:
      load_balance: KL(mean_gate || uniform) — encourages equal usage
      entropy: mean per-example gate entropy — minimize for sparser routing

    NOTE: Do NOT use L1 on softmax gates — always sums to 1, useless.
    Use entropy instead for sparsification.
    """
    B, N = gate.shape
    if target_prior is None:
        target_prior = torch.full((N,), 1.0 / N, device=gate.device, dtype=gate.dtype)

    # Load balance: KL divergence of batch-mean gate from uniform
    mean_gate = gate.mean(dim=0).clamp_min(1e-8)
    load_balance = torch.sum(mean_gate * (mean_gate.log() - target_prior.log()))

    # Entropy: negative entropy of per-example routing (minimize for sparsity)
    entropy = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=-1).mean()

    return load_balance, entropy


def diversity_loss(expert_mems, gate):
    """
    Mild diversity regularizer: penalize cosine similarity between
    expert memories weighted by gate. GPT-5.4 says set λ_div ≤ 1e-4
    or even 0 initially.
    """
    N = len(expert_mems)
    if N < 2:
        return torch.tensor(0.0, device=expert_mems[0].device)

    # Mean-pool each expert's memory
    means = [m.mean(dim=1) for m in expert_mems]  # list of (B, D)
    total_sim = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            cos = F.cosine_similarity(means[i], means[j], dim=-1).mean()
            total_sim = total_sim + cos
            count += 1

    return total_sim / count if count > 0 else torch.tensor(0.0, device=expert_mems[0].device)


# ============================================================
# Specialization diagnostics
# ============================================================
def compute_specialization_metrics(gate, domain_labels=None):
    """
    Measure specialization of the routing.

    Returns:
      n_eff: effective number of experts used (exp(H(mean_gate)))
      per_example_entropy: mean entropy of per-example routing
      mutual_info: MI(expert; domain) if domain labels provided
    """
    B, N = gate.shape

    # Effective number of experts
    mean_gate = gate.mean(dim=0).clamp_min(1e-8)
    H_mean = -(mean_gate * mean_gate.log()).sum()
    n_eff = torch.exp(H_mean).item()

    # Per-example entropy
    H_per = -(gate.clamp_min(1e-8) * gate.clamp_min(1e-8).log()).sum(dim=-1)
    per_example_entropy = H_per.mean().item()

    metrics = {
        'n_eff': n_eff,
        'per_example_entropy': per_example_entropy,
        'mean_gate': mean_gate.detach().cpu().tolist(),
    }

    # Mutual information if domain labels available
    if domain_labels is not None:
        # domain_labels: (B,) integer domain indices
        n_domains = domain_labels.max().item() + 1
        # Joint distribution p(expert, domain)
        joint = torch.zeros(N, n_domains, device=gate.device)
        for d in range(n_domains):
            mask = (domain_labels == d)
            if mask.sum() > 0:
                joint[:, d] = gate[mask].mean(dim=0)
        joint = joint / joint.sum().clamp_min(1e-8)

        p_expert = joint.sum(dim=1).clamp_min(1e-8)
        p_domain = joint.sum(dim=0).clamp_min(1e-8)

        mi = 0.0
        for e in range(N):
            for d in range(n_domains):
                if joint[e, d] > 1e-10:
                    mi += joint[e, d] * (joint[e, d].log() - p_expert[e].log() - p_domain[d].log())
        metrics['mutual_info'] = mi.item()

    return metrics


if __name__ == '__main__':
    # Quick test
    print("=== MoERM-Lite Architecture Test ===")

    # Test MoERM-Lite (full separate experts)
    moerm = MoERMLite(n_experts=4, d_solver=512, n_memory_slots_per_expert=32,
                      n_output_slots=32, llm_dim=4096, shared_core=False)
    total, trainable = moerm.count_params()
    print(f"MoERM-Lite (4 separate experts): {total:,} params ({total/1e6:.1f}M)")

    # Test shared-core variant
    moerm_shared = MoERMLite(n_experts=4, d_solver=512, n_memory_slots_per_expert=32,
                             n_output_slots=32, llm_dim=4096, shared_core=True)
    total_s, _ = moerm_shared.count_params()
    print(f"MoERM-Lite (shared core): {total_s:,} params ({total_s/1e6:.1f}M)")

    # Test forward pass
    x = torch.randn(2, 100, 4096, dtype=torch.bfloat16)

    moerm = moerm.to(dtype=torch.bfloat16)
    mem, gate, logits = moerm(x, K_inner=2, K_outer=1)
    print(f"Memory output: {mem.shape}")  # should be (2, 32, 4096)
    print(f"Gate: {gate.shape}, values: {gate[0].detach().tolist()}")

    # Test regularizers
    lb, ent = router_regularizers(gate)
    print(f"Load balance: {lb.item():.4f}, Entropy: {ent.item():.4f}")

    # Test specialization metrics
    metrics = compute_specialization_metrics(gate)
    print(f"N_eff: {metrics['n_eff']:.2f}, Per-example entropy: {metrics['per_example_entropy']:.4f}")

    print("\n=== All components working ===")
