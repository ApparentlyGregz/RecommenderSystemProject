# Results on the test cut

Commit `26dc76a`, 2026-11-27 09:21. Models: full, cl_noforget, nocl_forget, nocl_noforget. 1000 bootstrap resamples, seed 42.

Eligible students: 703 (102 with a failed topic not practised for > 14 days).

| Model | ndcg@10 | recall@20 | recall@5 | recall@10_old |
|---|---|---|---|---|
| Random (mean of seeds) | 0.6014 | 0.9935 | 0.3129 | 0.6421 |
| Most-popular (attempt frequency) | 0.7490 | 0.9941 | 0.4143 | 0.9586 |
| Global error rate | 0.7229 | 0.9952 | 0.4124 | 0.4467 |
| Personal error rate | 0.7164 | 0.9955 | 0.4083 | 0.5259 |
| Recent activity (answers in the last window) | 0.6542 | 0.9943 | 0.3213 | 0.4556 |
| Full model (NLGCL + forgetting) | 0.8046 | 0.9953 | 0.4667 | 0.5090 |
| NLGCL, no forgetting | 0.8004 | 0.9970 | 0.4502 | 0.7697 |
| LightGCN + forgetting | 0.7994 | 0.9955 | 0.4678 | 0.4601 |
| LightGCN | 0.8007 | 0.9971 | 0.4530 | 0.7697 |

## Gaps (paired bootstrap over students, 95% interval)

| Model | Reference | Metric | Gap | 95% CI | Significant |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | Random (mean of seeds) | ndcg@10 | +0.2032 | [+0.1931, +0.2133] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@20 | +0.0018 | [+0.0006, +0.0030] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@10_old | -0.1330 | [-0.2040, -0.0635] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | ndcg@10 | +0.0556 | [+0.0435, +0.0683] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@20 | +0.0012 | [-0.0010, +0.0035] | no |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@10_old | -0.4496 | [-0.5296, -0.3660] | yes |
| Full model (NLGCL + forgetting) | Global error rate | ndcg@10 | +0.0817 | [+0.0716, +0.0918] | yes |
| Full model (NLGCL + forgetting) | Global error rate | recall@20 | +0.0000 | [-0.0015, +0.0018] | no |
| Full model (NLGCL + forgetting) | Global error rate | recall@10_old | +0.0623 | [-0.0083, +0.1283] | no |
| Full model (NLGCL + forgetting) | Personal error rate | ndcg@10 | +0.0882 | [+0.0780, +0.0991] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | recall@20 | -0.0002 | [-0.0015, +0.0012] | no |
| Full model (NLGCL + forgetting) | Personal error rate | recall@10_old | -0.0169 | [-0.0891, +0.0564] | no |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | ndcg@10 | +0.1504 | [+0.1364, +0.1651] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | recall@20 | +0.0010 | [-0.0005, +0.0028] | no |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | recall@10_old | +0.0534 | [-0.0314, +0.1328] | no |
| NLGCL, no forgetting | Random (mean of seeds) | ndcg@10 | +0.1989 | [+0.1890, +0.2101] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@20 | +0.0035 | [+0.0020, +0.0050] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@10_old | +0.1276 | [+0.0564, +0.1896] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.0514 | [+0.0408, +0.0627] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@20 | +0.0029 | [+0.0005, +0.0054] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@10_old | -0.1889 | [-0.2629, -0.1218] | yes |
| NLGCL, no forgetting | Global error rate | ndcg@10 | +0.0774 | [+0.0662, +0.0876] | yes |
| NLGCL, no forgetting | Global error rate | recall@20 | +0.0017 | [+0.0000, +0.0037] | yes |
| NLGCL, no forgetting | Global error rate | recall@10_old | +0.3229 | [+0.2485, +0.4029] | yes |
| NLGCL, no forgetting | Personal error rate | ndcg@10 | +0.0840 | [+0.0732, +0.0951] | yes |
| NLGCL, no forgetting | Personal error rate | recall@20 | +0.0015 | [-0.0001, +0.0034] | no |
| NLGCL, no forgetting | Personal error rate | recall@10_old | +0.2438 | [+0.1577, +0.3317] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | ndcg@10 | +0.1461 | [+0.1331, +0.1588] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | recall@20 | +0.0027 | [+0.0010, +0.0047] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | recall@10_old | +0.3141 | [+0.2228, +0.4072] | yes |
| LightGCN + forgetting | Random (mean of seeds) | ndcg@10 | +0.1980 | [+0.1879, +0.2085] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@20 | +0.0020 | [+0.0007, +0.0033] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@10_old | -0.1820 | [-0.2549, -0.1105] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.0504 | [+0.0379, +0.0637] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@20 | +0.0014 | [-0.0009, +0.0039] | no |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@10_old | -0.4985 | [-0.5846, -0.4148] | yes |
| LightGCN + forgetting | Global error rate | ndcg@10 | +0.0765 | [+0.0659, +0.0870] | yes |
| LightGCN + forgetting | Global error rate | recall@20 | +0.0002 | [-0.0014, +0.0020] | no |
| LightGCN + forgetting | Global error rate | recall@10_old | +0.0133 | [-0.0590, +0.0829] | no |
| LightGCN + forgetting | Personal error rate | ndcg@10 | +0.0830 | [+0.0723, +0.0942] | yes |
| LightGCN + forgetting | Personal error rate | recall@20 | -0.0000 | [-0.0013, +0.0012] | no |
| LightGCN + forgetting | Personal error rate | recall@10_old | -0.0658 | [-0.1420, +0.0063] | no |
| LightGCN + forgetting | Recent activity (answers in the last window) | ndcg@10 | +0.1452 | [+0.1310, +0.1604] | yes |
| LightGCN + forgetting | Recent activity (answers in the last window) | recall@20 | +0.0011 | [-0.0004, +0.0029] | no |
| LightGCN + forgetting | Recent activity (answers in the last window) | recall@10_old | +0.0045 | [-0.0866, +0.0871] | no |
| LightGCN | Random (mean of seeds) | ndcg@10 | +0.1993 | [+0.1892, +0.2101] | yes |
| LightGCN | Random (mean of seeds) | recall@20 | +0.0036 | [+0.0021, +0.0052] | yes |
| LightGCN | Random (mean of seeds) | recall@10_old | +0.1277 | [+0.0586, +0.1933] | yes |
| LightGCN | Most-popular (attempt frequency) | ndcg@10 | +0.0517 | [+0.0410, +0.0633] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@20 | +0.0030 | [+0.0007, +0.0055] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@10_old | -0.1889 | [-0.2606, -0.1224] | yes |
| LightGCN | Global error rate | ndcg@10 | +0.0778 | [+0.0668, +0.0882] | yes |
| LightGCN | Global error rate | recall@20 | +0.0018 | [+0.0001, +0.0038] | yes |
| LightGCN | Global error rate | recall@10_old | +0.3230 | [+0.2492, +0.4047] | yes |
| LightGCN | Personal error rate | ndcg@10 | +0.0843 | [+0.0735, +0.0951] | yes |
| LightGCN | Personal error rate | recall@20 | +0.0016 | [-0.0000, +0.0035] | no |
| LightGCN | Personal error rate | recall@10_old | +0.2438 | [+0.1566, +0.3328] | yes |
| LightGCN | Recent activity (answers in the last window) | ndcg@10 | +0.1465 | [+0.1337, +0.1597] | yes |
| LightGCN | Recent activity (answers in the last window) | recall@20 | +0.0028 | [+0.0010, +0.0049] | yes |
| LightGCN | Recent activity (answers in the last window) | recall@10_old | +0.3141 | [+0.2260, +0.4099] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | ndcg@10 | +0.0043 | [-0.0034, +0.0114] | no |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@20 | -0.0017 | [-0.0029, -0.0006] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@10_old | -0.2607 | [-0.3329, -0.1927] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | ndcg@10 | +0.0052 | [+0.0027, +0.0078] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@20 | -0.0002 | [-0.0005, +0.0000] | no |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@10_old | +0.0489 | [+0.0171, +0.0837] | yes |
| Full model (NLGCL + forgetting) | LightGCN | ndcg@10 | +0.0039 | [-0.0040, +0.0117] | no |
| Full model (NLGCL + forgetting) | LightGCN | recall@20 | -0.0018 | [-0.0031, -0.0006] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@10_old | -0.2607 | [-0.3339, -0.1946] | yes |

## Sanity floor (Week 3 Q2)

Beats random and most-popular on NDCG@10 and Recall@20, with the 95% interval of every gap above 0.

- Full model (NLGCL + forgetting): **no**
- NLGCL, no forgetting: **yes**
- LightGCN + forgetting: **no**
- LightGCN: **yes**

## Learned forgetting parameters

beta > 0 means a topic gets riskier the longer it has not been practised.

| Model | beta | never-seen value | half-life median (days) | p10 | p90 |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | -0.0245 | +0.1370 | 6.88 | 5.92 | 7.87 |
| LightGCN + forgetting | -0.0239 | +0.1180 | 6.67 | 5.96 | 7.70 |
