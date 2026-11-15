"""Check: gradient flows to a learned per-notion tau through degree-normalised sparse propagation."""
import torch

torch.manual_seed(0)
n_s, n_n, d = 50, 10, 8
edges = torch.randint(0, n_s, (200,)), torch.randint(0, n_n, (200,))
dt = torch.rand(200) * 30  # days since last error

log_tau = torch.nn.Parameter(torch.full((n_n,), 2.0))  # tau = exp(log_tau) keeps tau > 0
emb = torch.nn.Parameter(torch.randn(n_s + n_n, d) * 0.1)

w = torch.exp(-dt / torch.exp(log_tau)[edges[1]])
row = torch.cat([edges[0], edges[1] + n_s])
col = torch.cat([edges[1] + n_s, edges[0]])
val = torch.cat([w, w])
deg = torch.zeros(n_s + n_n).index_add(0, row, val)  # weighted degree
norm_val = val / torch.sqrt(deg[row] * deg[col])
A = torch.sparse_coo_tensor(torch.stack([row, col]), norm_val, (n_s + n_n, n_s + n_n)).coalesce()

h = emb
layers = [h]
for _ in range(2):
    h = torch.sparse.mm(A, h)
    layers.append(h)
out = torch.stack(layers).mean(0)
out.pow(2).sum().backward()
print("grad on log_tau finite:", bool(torch.isfinite(log_tau.grad).all()), "| non-zero:", int((log_tau.grad != 0).sum()), "/", n_n)
