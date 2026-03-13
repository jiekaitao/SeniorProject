"""Benchmark forward+backward speed for all RNAformer_TRM variants."""
import torch
import time
import sys
sys.path.insert(0, '/mnt/Data/GitHub/SeniorProject')
sys.path.insert(0, '/mnt/Data/GitHub/SeniorProject/RNAformer')

from RNAformer_TRM.tests.conftest import make_tiny_config, make_model
from RNAformer_TRM.losses.rna_act_loss import RNAACTLoss


def bench_config(**kw):
    return make_tiny_config(model_dim=64, num_head=4, n_layers=4, max_len=64, **kw)


class CallCounter:
    def __init__(self, original_fn):
        self.original_fn = original_fn
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.original_fn(*args, **kwargs)


def main():
    B, L = 4, 32
    src_seq = torch.randint(0, 5, (B, L))
    length = torch.full((B,), L)
    pdb_sample = torch.ones(B, 1)
    N_ITERS = 20

    configs = {
        'Baseline (no cycling)': dict(),
        'Cycling=4 (original)': dict(cycling=4, cgar_enabled=False, input_reinjection=False),
        'CGAR cycling=6': dict(cycling=6, cgar_enabled=True, input_reinjection=True),
        'Dual-state H=2,L=3': dict(dual_state=True, H_cycles=2, L_cycles=3),
        'Dual-state H=3,L=6 + ACT': dict(dual_state=True, H_cycles=3, L_cycles=6, act_enabled=True),
    }

    print(f'Benchmarking {N_ITERS} forward+backward passes (B={B}, L={L}, dim=64, layers=4)')
    print(f'{"Config":<35} {"Fwd+Bwd (ms)":<15} {"Relative":<10} {"Stack calls":<12}')
    print('-' * 75)

    baseline_time = None
    for name, kw in configs.items():
        cfg = bench_config(**kw)
        model = make_model(cfg)
        model.train()
        if hasattr(model, 'set_curriculum_depth'):
            model.set_curriculum_depth(1.0)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            result = model(src_seq, length, pdb_sample, max_cycle=kw.get('cycling', 0))
            if len(result) == 3:
                loss = result[0].sum() + result[2].sum()
            else:
                loss = result[0].sum()
            loss.backward()
            optimizer.step()

        # Count stack calls (single pass)
        counter = CallCounter(model.RNAformer.forward)
        model.RNAformer.forward = counter
        with torch.no_grad():
            model.eval()
            model(src_seq, length, pdb_sample, max_cycle=kw.get('cycling', 0))
            model.train()
        calls_per_iter = counter.count
        model.RNAformer.forward = counter.original_fn

        # Timed run
        start = time.perf_counter()
        for i in range(N_ITERS):
            optimizer.zero_grad()
            result = model(src_seq, length, pdb_sample, max_cycle=kw.get('cycling', 0))
            if len(result) == 3:
                loss = result[0].sum() + result[2].sum()
            else:
                loss = result[0].sum()
            loss.backward()
            optimizer.step()
        elapsed = (time.perf_counter() - start) / N_ITERS * 1000

        if baseline_time is None:
            baseline_time = elapsed

        rel = elapsed / baseline_time
        print(f'{name:<35} {elapsed:>8.1f} ms    {rel:>5.2f}x      {calls_per_iter}')

    print()
    print("Note: Cycling variants use random cycle count during training,")
    print("so actual training speed varies. Stack calls shown are for eval (full depth).")


if __name__ == '__main__':
    main()
