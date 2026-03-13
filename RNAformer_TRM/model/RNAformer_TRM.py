import math

import torch
import torch.nn as nn

from RNAformer_TRM.module.embedding import EmbedSequence2Matrix
from RNAformer_TRM.model.RNAformer_stack import RNAformerStack


class RiboFormerTRM(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.model_dim = config.model_dim

        # Phase 1: CGAR config
        self.cgar_enabled = getattr(config, "cgar_enabled", False)
        self.input_reinjection = getattr(config, "input_reinjection", False)
        self.supervision_decay = getattr(config, "supervision_decay", 0.7)

        # Phase 2: Dual-state config
        self.dual_state = getattr(config, "dual_state", False)
        self.H_cycles = getattr(config, "H_cycles", 1)
        self.L_cycles = getattr(config, "L_cycles", 1)

        # Phase 3: ACT config
        self.act_enabled = getattr(config, "act_enabled", False)

        # Cycling initialization (same as original)
        if hasattr(config, "cycling") and config.cycling:
            self.initialize_cycling(config.cycling)
        else:
            self.cycling = False

        self.seq2mat_embed = EmbedSequence2Matrix(config)
        self.RNAformer = RNAformerStack(config)

        if not hasattr(config, "pdb_flag") or config.pdb_flag:
            self.pdf_embedding = nn.Linear(1, config.model_dim, bias=True)
            self.use_pdb = True
        else:
            self.use_pdb = False

        if not hasattr(config, "binary_output") or config.binary_output:
            self.output_mat = nn.Linear(config.model_dim, 1, bias=True)
        else:
            self.output_mat = nn.Linear(config.model_dim, 2, bias=False)

        # Phase 2: Dual-state LayerNorms
        if self.dual_state:
            self.recycle_zH_norm = nn.LayerNorm(self.model_dim, elementwise_affine=True)
            self.recycle_zL_norm = nn.LayerNorm(self.model_dim, elementwise_affine=True)

        # Phase 3: ACT q_head
        if self.act_enabled:
            self.q_head = nn.Sequential(
                nn.Linear(config.model_dim, 1),
            )

        self.initialize(initializer_range=config.initializer_range)

    def initialize(self, initializer_range):
        nn.init.normal_(self.output_mat.weight, mean=0.0, std=initializer_range)

    def initialize_cycling(self, cycle_steps):
        import random
        self.cycling = True
        self.cycle_steps = cycle_steps
        self.base_cycle_steps = cycle_steps
        self.current_cycle_steps = cycle_steps
        self.recycle_pair_norm = nn.LayerNorm(self.model_dim, elementwise_affine=True)
        self.trng = torch.Generator()
        self.trng.manual_seed(random.randint(1, 10000))

    def set_curriculum_depth(self, progress: float):
        """Phase 1: CGAR progressive cycling depth.

        3-stage curriculum that gradually increases the number of recycle steps
        as training progresses.

        Args:
            progress: Training progress in [0, 1].
        """
        if not self.cgar_enabled or not self.cycling:
            return

        if progress < 0.3:
            self.current_cycle_steps = math.ceil(self.base_cycle_steps / 3)
        elif progress < 0.6:
            self.current_cycle_steps = math.ceil(2 * self.base_cycle_steps / 3)
        else:
            self.current_cycle_steps = self.base_cycle_steps

    def make_pair_mask(self, src, src_len):
        encode_mask = torch.arange(src.shape[1], device=src.device).expand(src.shape[:2]) < src_len.unsqueeze(1)
        pair_mask = encode_mask[:, None, :] * encode_mask[:, :, None]
        assert isinstance(pair_mask, torch.BoolTensor) or isinstance(pair_mask, torch.cuda.BoolTensor)
        return torch.bitwise_not(pair_mask)

    @torch.no_grad()
    def cycle_riboformer(self, pair_act, pair_mask):
        latent = self.RNAformer(pair_act=pair_act, pair_mask=pair_mask, cycle_infer=True)
        return latent.detach()

    def _get_n_cycles(self, max_cycle):
        """Determine the number of cycles for this forward pass.

        During training: random number in [0, effective_max], where
        effective_max respects the CGAR curriculum depth.

        During eval: always use full base_cycle_steps (ignoring curriculum).
        """
        if self.training:
            effective_max = self.current_cycle_steps if self.cgar_enabled else max_cycle
            n_cycles = torch.randint(0, effective_max + 1, [1])
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                n_cycles = n_cycles.to(torch.int64).cuda()
                tensor_list = [torch.zeros(1, dtype=torch.int64).cuda()
                               for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(tensor_list, n_cycles)
                n_cycles = tensor_list[0]
            n_cycles = n_cycles.item()
        else:
            n_cycles = self.base_cycle_steps if self.cgar_enabled else self.cycle_steps
        return n_cycles

    def forward_cgar_cycling(self, pair_latent, pair_mask, input_embed, max_cycle):
        """Phase 1: CGAR progressive cycling with optional input re-injection.

        Replaces the inline cycling logic from the original forward method.
        """
        n_cycles = self._get_n_cycles(max_cycle)

        cyc_latent = torch.zeros_like(pair_latent)
        for n in range(n_cycles - 1):
            res_latent = pair_latent.detach() + self.recycle_pair_norm(cyc_latent.detach()).detach()
            if self.input_reinjection:
                res_latent = res_latent + input_embed.detach()
            cyc_latent = self.cycle_riboformer(res_latent, pair_mask.detach())

        pair_latent = pair_latent + self.recycle_pair_norm(cyc_latent.detach())
        if self.input_reinjection:
            pair_latent = pair_latent + input_embed

        latent = self.RNAformer(pair_act=pair_latent, pair_mask=pair_mask, cycle_infer=False)
        return latent

    def forward_dual_state(self, pair_latent, pair_mask, input_embed):
        """Phase 2: Dual-state z_H / z_L recursion.

        Uses the SAME RNAformerStack for both states (weight tying, zero
        parameter overhead beyond two LayerNorms).

        CGAR integration: when cgar_enabled and training, scale H_cycles and
        L_cycles by the curriculum ratio (current / base).
        """
        B, L1, L2, D = pair_latent.shape

        z_H = torch.zeros_like(pair_latent)
        z_L = torch.zeros_like(pair_latent)

        # Determine effective cycle counts
        h_cycles = self.H_cycles
        l_cycles = self.L_cycles
        if self.cgar_enabled and self.cycling and self.training:
            ratio = self.current_cycle_steps / self.base_cycle_steps
            h_cycles = max(1, math.ceil(self.H_cycles * ratio))
            l_cycles = max(1, math.ceil(self.L_cycles * ratio))

        if not self.training and self.cgar_enabled and self.cycling:
            # Eval: always use full cycle counts
            h_cycles = self.H_cycles
            l_cycles = self.L_cycles

        for h in range(h_cycles):
            is_last_h = (h == h_cycles - 1)

            # L-cycle updates
            for l_step in range(l_cycles):
                zl_input = self.recycle_zL_norm(z_L) + self.recycle_zH_norm(z_H) + input_embed
                if is_last_h:
                    z_L = self.RNAformer(pair_act=zl_input, pair_mask=pair_mask, cycle_infer=False)
                else:
                    with torch.no_grad():
                        z_L = self.RNAformer(pair_act=zl_input.detach(), pair_mask=pair_mask, cycle_infer=True).detach()

            # H-cycle update
            zh_input = self.recycle_zH_norm(z_H) + self.recycle_zL_norm(z_L) + pair_latent
            if is_last_h:
                z_H = self.RNAformer(pair_act=zh_input, pair_mask=pair_mask, cycle_infer=False)
            else:
                with torch.no_grad():
                    z_H = self.RNAformer(pair_act=zh_input.detach(), pair_mask=pair_mask, cycle_infer=True).detach()

        return z_H

    def forward(self, src_seq, src_len, pdb_sample, max_cycle=0):
        pair_mask = self.make_pair_mask(src_seq, src_len)
        pair_latent = self.seq2mat_embed(src_seq)

        if self.use_pdb:
            pair_latent = pair_latent + self.pdf_embedding(pdb_sample)[:, None, None, :]

        input_embed = pair_latent.clone()

        if self.dual_state:
            latent = self.forward_dual_state(pair_latent, pair_mask, input_embed)
        elif self.cycling:
            latent = self.forward_cgar_cycling(pair_latent, pair_mask, input_embed, max_cycle)
        else:
            latent = self.RNAformer(pair_act=pair_latent, pair_mask=pair_mask, cycle_infer=False)

        logits = self.output_mat(latent)

        if self.act_enabled:
            pooled = latent.mean(dim=(1, 2))  # [B, D]
            q_halt_logit = self.q_head(pooled).squeeze(-1)  # [B]
            return logits, pair_mask, q_halt_logit

        return logits, pair_mask
