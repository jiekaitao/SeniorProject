"""
Scaled Grid Reachability: bigger grids, more complex, prove K-scaling scales.
32x32 grid with walls, 2D flood fill. Local attention window=3.
K=1 sees 7x7, K=4 sees 25x25, K=8 sees full 32x32.
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, math, time

device = torch.device('cuda')
N = 32  # 32x32 grid
BATCH = 128

def make_batch(bs):
    # 2D grid: 0=open, 1=wall. Top-left always open.
    worlds = (torch.rand(bs, N, N, device=device) > 0.7).float()  # 30% walls
    worlds[:, 0, 0] = 0
    # Ground truth: BFS reachability from (0,0)
    reach = torch.zeros(bs, N, N, device=device)
    reach[:, 0, 0] = 1
    for _ in range(N * 2):  # enough iterations for full propagation
        padded = F.pad(reach, (1,1,1,1))
        neighbors = (padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] +
                     padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:])
        reach = ((reach + (neighbors > 0).float()) > 0).float() * (1 - worlds)
    return worlds.view(bs, N*N), reach.view(bs, N*N)

class Local2DBlock(nn.Module):
    """2D local convolution-style block. Each cell sees 3x3 neighborhood."""
    def __init__(self, d=64):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv2d(d, d*2, kernel_size=3, padding=1, groups=1)
        self.proj = nn.Conv2d(d*2, d, kernel_size=1)
    def forward(self, x, N):
        B, T, D = x.shape
        h = self.norm(x).view(B, N, N, D).permute(0, 3, 1, 2)
        h = self.proj(F.gelu(self.conv(h)))
        h = h.permute(0, 2, 3, 1).view(B, T, D)
        return x + h

class Grid2DSolver(nn.Module):
    def __init__(self, d=64, n_blocks=2):
        super().__init__()
        self.embed = nn.Linear(1, d)
        self.blocks = nn.ModuleList([Local2DBlock(d) for _ in range(n_blocks)])
        self.head = nn.Linear(d, 1)
    def forward(self, worlds, K, N=32):
        x = self.embed(worlds.unsqueeze(-1))
        for _ in range(K):
            for blk in self.blocks:
                x = blk(x, N)
        return self.head(x).squeeze(-1)

model = Grid2DSolver(d=64, n_blocks=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f'2D Grid {N}x{N}, params: {sum(p.numel() for p in model.parameters()):,}', flush=True)
print(f'K=1: 3x3 receptive field. K=4: ~9x9. K=8: ~17x17. K=16: full {N}x{N}', flush=True)

t0 = time.time()
for step in range(10000):
    worlds, reach = make_batch(BATCH)
    K = random.choice([2, 4, 8, 16])
    pred = model(worlds, K)
    loss = F.binary_cross_entropy_with_logits(pred, reach)
    loss.backward(); optimizer.step(); optimizer.zero_grad()

    if (step+1) % 500 == 0:
        model.eval()
        w, r = make_batch(2000)
        results = {}
        with torch.no_grad():
            for Ke in [1, 2, 4, 8, 16, 32]:
                p = (model(w, Ke) > 0).float()
                results[Ke] = (p == r).float().mean().item()
        parts = [f'K={k}={v:.4f}' for k,v in sorted(results.items())]
        print(f'  step {step+1:5d} | loss={loss.item():.4f} | {" | ".join(parts)} | {time.time()-t0:.0f}s', flush=True)
        if results[2] > results[1] + 0.01 and results[4] > results[2] + 0.01:
            print(f'  >>> MONOTONE K-SCALING <<<', flush=True)
        model.train()

print(f'\n=== Final ===', flush=True)
model.eval()
w, r = make_batch(5000)
with torch.no_grad():
    for Ke in [1, 2, 4, 8, 16, 32]:
        p = (model(w, Ke) > 0).float()
        acc = (p == r).float().mean().item()
        print(f'  K={Ke:2d}: accuracy={acc:.4f}', flush=True)
