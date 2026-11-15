"""Toy simulation: does training on 'failed in the train window' recover a positive forgetting effect beta?

Ground truth: the probability of failing a notion rises with the time since it was last practised.
Version A (plan as written): positives = notions failed in the train window, negatives = random other notions,
Delta-t measured at the end of the train window (never practised -> capped at 60 days).
Version B (proposed fix): features from history before a cutoff, labels = failed in the next window,
negatives = notions attempted but not failed in that window.
"""
import numpy as np
import torch

rng = np.random.default_rng(0)
torch.manual_seed(0)
U, N, T, CUT_A, CUT_B = 3000, 100, 60.0, 48.0, 36.0
pop = rng.dirichlet(np.ones(N) * 0.5)          # exposure skew, like Santa's free parts
skill = rng.normal(0, 1, (U, N))                 # latent per-student weakness
TAU_TRUE, FORGET = 7.0, 2.0

events = []  # (u, n, t, failed)
for u in range(U):
    k = rng.integers(20, 80)
    notions = rng.choice(N, size=k, p=pop)
    times = np.sort(rng.uniform(0, T, size=k))
    last = {}
    for n, t in zip(notions, times):
        gap = t - last[n] if n in last else 30.0
        logit = -1.0 + 0.8 * skill[u, n] + FORGET * (1 - np.exp(-gap / TAU_TRUE))
        failed = rng.random() < 1 / (1 + np.exp(-logit))
        events.append((u, n, t, failed))
        last[n] = t
ev = np.array(events, dtype=float)


def build(cut, label_end, version):
    hist = ev[ev[:, 2] < cut]
    last_prac = np.full((U, N), np.nan)
    for u, n, t, _ in hist:
        last_prac[int(u), int(n)] = t
    dt = np.where(np.isnan(last_prac), 60.0, cut - last_prac)
    if version == "A":
        lab = hist  # labels come from the same window as the features
        pos = {(int(u), int(n)) for u, n, _, f in lab if f}
        pairs = []
        for u, n in pos:
            neg = rng.integers(N)
            while (u, neg) in pos:
                neg = rng.integers(N)
            pairs.append((u, n, neg))
    else:
        lab = ev[(ev[:, 2] >= cut) & (ev[:, 2] < label_end)]
        att, fail = {}, {}
        for u, n, _, f in lab:
            att.setdefault(int(u), set()).add(int(n))
            if f:
                fail.setdefault(int(u), set()).add(int(n))
        pairs = []
        for u, fs in fail.items():
            negs = list(att[u] - fs)
            for n in fs:
                if negs:
                    pairs.append((u, n, negs[rng.integers(len(negs))]))
    return np.array(pairs), torch.tensor(dt, dtype=torch.float32)


def fit(pairs, dt):
    eu = torch.nn.Parameter(torch.randn(U, 16) * 0.1)
    en = torch.nn.Parameter(torch.randn(N, 16) * 0.1)
    beta = torch.nn.Parameter(torch.tensor(0.0))
    log_tau = torch.nn.Parameter(torch.full((N,), np.log(7.0), dtype=torch.float32))
    opt = torch.optim.Adam([eu, en, beta, log_tau], lr=0.02)
    p = torch.tensor(pairs, dtype=torch.long)
    for _ in range(300):
        u, i, j = p[:, 0], p[:, 1], p[:, 2]
        tau = torch.exp(log_tau)

        def s(n):
            return (eu[u] * en[n]).sum(1) + beta * (1 - torch.exp(-dt[u, n] / tau[n]))

        loss = -torch.nn.functional.logsigmoid(s(i) - s(j)).mean() + 1e-4 * (eu.pow(2).sum() + en.pow(2).sum()) / len(p)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return beta.item()


pa, dta = build(CUT_A, CUT_A, "A")
pb, dtb = build(CUT_B, CUT_A, "B")
print(f"True effect: failing gets MORE likely when a notion was not practised for a long time (beta should be > 0)")
print(f"Version A (plan as written) learned beta = {fit(pa, dta):+.2f}  on {len(pa)} pairs")
print(f"Version B (history -> next window, attempted negatives) learned beta = {fit(pb, dtb):+.2f}  on {len(pb)} pairs")
