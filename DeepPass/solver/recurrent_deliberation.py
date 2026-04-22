"""
Recurrent Deliberation Controller — The thesis architecture.

A frozen LLM with a learned recurrent control interface that allocates
extra latent compute only when needed.

NOT a task-specific solver. A GENERAL iterative reasoning interface:
  Round 1: Controller writes initial thought tokens → frozen LM processes
  Round 2: Controller reads LM's hidden states → writes UPDATED thoughts → LM re-processes
  ...
  The controller learns WHEN and HOW to help the frozen LM think harder.

Key design choices (from GPT-5.4 consultation + literature):
  1. Thought slots are sparse vocab superpositions (native to LM manifold)
  2. Controller reads hidden states at tapped layers + logits + uncertainty
  3. Verifier head predicts whether current answer is correct
  4. Progress loss ensures later rounds don't degrade

References: COCONUT, Recurrent-Depth, MoR, Dr.LLM, StateLM, ASM
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


class RecurrentDeliberation(nn.Module):
    """
    Recurrent Deliberation Controller over a frozen LLM backbone.

    The controller maintains latent state z across rounds:
      1. z → thought embeddings (sparse vocab superposition)
      2. Frozen LM processes [prompt | thoughts | answer_prefix]
      3. Controller reads hidden states at tapped layers + answer logits
      4. z is updated for next round

    The controller does NOT output the answer — it outputs HOW the frozen
    model should think next.
    """
    def __init__(self, frozen_llm, d_state=512, n_slots=8,
                 tapped_layers=(8, 16, 24), topk_vocab=64):
        super().__init__()
        self.frozen_llm = frozen_llm
        self.tapped_layers = set(tapped_layers)
        self.tapped_list = sorted(tapped_layers)
        self.n_slots = n_slots
        self.topk_vocab = topk_vocab

        # Frozen LM properties (handle Gemma3/4 nested config)
        cfg = frozen_llm.config
        if hasattr(cfg, 'text_config'):
            self.d_model = cfg.text_config.hidden_size
            vocab_size = cfg.text_config.vocab_size
        else:
            self.d_model = cfg.hidden_size
            vocab_size = cfg.vocab_size

        # Freeze the backbone
        for p in frozen_llm.parameters():
            p.requires_grad = False

        # Initial controller state
        self.z0 = nn.Parameter(torch.randn(1, n_slots, d_state) * 0.02)

        # --- READ: hidden states + logits → feature vector ---
        # Per tapped layer: pool over sequence → d_model
        # Think slot hidden states: n_slots * d_model
        # Answer logits over choices: 4
        # Uncertainty: entropy + margin = 2
        n_tap = len(tapped_layers)
        read_dim = n_tap * self.d_model + n_slots * self.d_model + 4 + 2
        self.read_proj = nn.Sequential(
            nn.Linear(read_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, n_slots * d_state),
        )

        # --- STATE UPDATE: residual update on z ---
        self.state_norm = RMSNorm(d_state)
        self.state_gate = nn.Parameter(torch.tensor(0.1))  # Start small

        # --- WRITE: z → sparse vocab superposition → thought embeddings ---
        self.to_vocab_logits = nn.Linear(d_state, vocab_size, bias=False)
        # Initialize small so initial thoughts are near-zero
        nn.init.normal_(self.to_vocab_logits.weight, std=0.01)

        # --- VERIFIER: predict whether current answer is correct ---
        self.verifier = nn.Sequential(
            nn.Linear(n_tap * self.d_model + n_slots * self.d_model + 4, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

    def latent_to_thought_embs(self, z):
        """Convert controller state → thought embeddings as sparse vocab superpositions."""
        # z: (B, S, d_state)
        E = self.frozen_llm.model.embed_tokens.weight  # (V, d_model)
        logits = self.to_vocab_logits(z)  # (B, S, V)

        # Top-k sparse selection
        vals, idx = logits.topk(self.topk_vocab, dim=-1)  # (B, S, K)
        probs = F.softmax(vals, dim=-1)  # (B, S, K)

        # Weighted sum of vocab embeddings
        chosen_embs = E[idx]  # (B, S, K, d_model)
        thought_embs = (probs.unsqueeze(-1) * chosen_embs).sum(dim=-2)  # (B, S, d_model)
        return thought_embs

    def forward_frozen_round(self, prompt_emb, thought_emb, answer_emb):
        """Run one round through the frozen decoder with thought slots."""
        lm_model = self.frozen_llm.model

        # Assemble: [prompt | THINK slots | answer_prefix]
        dec_input = torch.cat([prompt_emb, thought_emb, answer_emb], dim=1)

        T = dec_input.shape[1]
        pos_ids = torch.arange(T, device=dec_input.device).unsqueeze(0)
        pos_emb = lm_model.rotary_emb(dec_input, pos_ids)

        # Track positions of think slots
        p_len = prompt_emb.shape[1]
        t_len = thought_emb.shape[1]
        think_slice = slice(p_len, p_len + t_len)

        h = dec_input
        tapped_pools = []

        for i, layer in enumerate(lm_model.layers):
            h = layer(h, position_embeddings=pos_emb)
            if i in self.tapped_layers:
                # Pool over full sequence for this tapped layer
                tapped_pools.append(h.mean(dim=1))  # (B, d_model)

        h = lm_model.norm(h)
        logits = self.frozen_llm.lm_head(h)

        # Extract think slot hidden states
        think_h = h[:, think_slice]  # (B, n_slots, d_model)

        return logits, think_h, tapped_pools

    def build_features(self, think_h, tapped_pools, choice_logits):
        """Build feature vector from frozen LM's output."""
        B = think_h.shape[0]
        dtype = think_h.dtype  # match frozen LM dtype (bf16)

        # Answer distribution features (compute in float, cast back)
        probs = choice_logits.float().softmax(dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        top2 = probs.topk(2, dim=-1).values
        margin = top2[:, :1] - top2[:, 1:2]

        # Concatenate all features, cast to model dtype
        feat = torch.cat(
            [think_h.flatten(1)] +
            tapped_pools +
            [probs.to(dtype), entropy.to(dtype), margin.to(dtype)],
            dim=-1
        )
        return feat

    def forward(self, prompt_emb, answer_emb, choice_ids, rounds=2):
        """
        Full deliberation loop.

        Args:
            prompt_emb: (B, P, d_model) frozen embeddings of prompt
            answer_emb: (B, A, d_model) frozen embeddings of "\nAnswer:" prefix
            choice_ids: tensor of token IDs for A/B/C/D
            rounds: number of deliberation rounds

        Returns:
            all_choice_logits: list of (B, 4) per round
            all_verify: list of (B, 1) per round
        """
        B = prompt_emb.shape[0]
        z = self.z0.expand(B, -1, -1).clone()  # (B, n_slots, d_state)

        all_choice_logits = []
        all_verify = []

        for r in range(rounds):
            # 1. Write: z → thought embeddings
            thought_emb = self.latent_to_thought_embs(z)  # (B, n_slots, d_model)

            # 2. Run frozen decoder
            logits, think_h, tapped_pools = self.forward_frozen_round(
                prompt_emb, thought_emb.to(prompt_emb.dtype), answer_emb
            )

            # 3. Extract answer logits for A/B/C/D
            # Answer position: last token in the sequence
            ans_logits = logits[:, -1, choice_ids]  # (B, 4)
            all_choice_logits.append(ans_logits)

            # 4. Build features from frozen LM output
            feat = self.build_features(think_h, tapped_pools, ans_logits)

            # 5. Verifier: is current answer correct?
            verify_probs = ans_logits.float().softmax(dim=-1).to(think_h.dtype)
            verify_feat = torch.cat(
                [think_h.flatten(1)] + tapped_pools + [verify_probs],
                dim=-1
            )
            verify = self.verifier(verify_feat)  # (B, 1)
            all_verify.append(verify)

            # 6. Update controller state (except last round)
            if r < rounds - 1:
                delta = self.read_proj(feat).view(B, self.n_slots, -1)
                z = self.state_norm(z + self.state_gate * delta)

        return all_choice_logits, all_verify

    def compute_loss(self, all_choice_logits, all_verify, answer_labels,
                     lambda_v=0.5, lambda_p=0.1, delta_p=0.1):
        """
        Compute the full deliberation loss.

        L = CE(final_answer, y)
            + λ_v * Σ BCE(v_r, correct_r)
            + λ_p * max(0, CE(final) - CE(first) + δ)
        """
        rounds = len(all_choice_logits)
        B = answer_labels.shape[0]

        # Final answer CE (compute in float for stability)
        final_ce = F.cross_entropy(all_choice_logits[-1].float(), answer_labels)

        # Verifier loss: predict whether each round's argmax matches truth
        verify_loss = torch.tensor(0.0, device=answer_labels.device, dtype=torch.float32)
        for r in range(rounds):
            pred_correct = (all_choice_logits[r].argmax(dim=-1) == answer_labels).float()
            verify_loss = verify_loss + F.binary_cross_entropy_with_logits(
                all_verify[r].float().squeeze(-1), pred_correct
            )
        verify_loss = verify_loss / rounds

        # Progress loss: round 2 should not be worse than round 1
        progress_loss = torch.tensor(0.0, device=answer_labels.device, dtype=torch.float32)
        if rounds > 1:
            first_ce = F.cross_entropy(all_choice_logits[0].float(), answer_labels)
            progress_loss = F.relu(final_ce - first_ce + delta_p)

        total = final_ce + lambda_v * verify_loss + lambda_p * progress_loss
        return total, {
            'final_ce': final_ce.item(),
            'verify_loss': verify_loss.item(),
            'progress_loss': progress_loss.item(),
        }

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
