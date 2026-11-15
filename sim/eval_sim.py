"""Toy simulation: what does 'notions failed in the test window' actually reward?

Same generator as beta_sim.py. We compare an ORACLE that knows each student's true risk of failing every notion
(but not which notions the app will serve next) with MOST-POPULAR, under two relevance definitions:
  M1 (plan): rank all notions, relevant = failed at least once in the test window.
  M2 (fix):  rank only notions the student attempted in the test window, relevant = failed among those.
"""
import numpy as np

rng = np.random.default_rng(0)
U, N, T, CUT = 3000, 100, 60.0, 48.0
pop = rng.dirichlet(np.ones(N) * 0.5)
skill = rng.normal(0, 1, (U, N))
TAU, FORGET = 7.0, 2.0

hist_fail = np.zeros(N)
last = np.full((U, N), np.nan)
test_att = [set() for _ in range(U)]
test_fail = [set() for _ in range(U)]
for u in range(U):
    k = rng.integers(20, 80)
    notions = rng.choice(N, size=k, p=pop)
    times = np.sort(rng.uniform(0, T, size=k))
    seen = {}
    for n, t in zip(notions, times):
        gap = t - seen[n] if n in seen else 30.0
        logit = -1.0 + 0.8 * skill[u, n] + FORGET * (1 - np.exp(-gap / TAU))
        failed = rng.random() < 1 / (1 + np.exp(-logit))
        if t < CUT:
            hist_fail[n] += failed
            last[u, n] = t
        else:
            test_att[u].add(n)
            if failed:
                test_fail[u].add(n)
        seen[n] = t


def ndcg10(ranked, rel):
    dcg = sum(1 / np.log2(i + 2) for i, n in enumerate(ranked[:10]) if n in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(rel))))
    return dcg / idcg


res = {"oracle": [[], []], "mostpop": [[], []]}
for u in range(U):
    if not test_fail[u]:
        continue
    gap = np.where(np.isnan(last[u]), 30.0, CUT - last[u])
    risk = -1.0 + 0.8 * skill[u] + FORGET * (1 - np.exp(-gap / TAU))
    scores = {"oracle": risk, "mostpop": hist_fail}
    for name, sc in scores.items():
        order = list(np.argsort(-sc))
        res[name][0].append(ndcg10(order, test_fail[u]))
        cand = [n for n in order if n in test_att[u]]
        res[name][1].append(ndcg10(cand, test_fail[u]))

for name, (m1, m2) in res.items():
    print(f"{name:8s}  M1 (plan) NDCG@10 = {np.mean(m1):.3f}   M2 (attempted only) NDCG@10 = {np.mean(m2):.3f}")
