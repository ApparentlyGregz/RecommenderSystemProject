# Results on the test cut

Commit `3593cb7`, 2026-11-22 18:29. Models: full, cl_noforget, nocl_forget, nocl_noforget. 1000 bootstrap resamples, seed 42.

Eligible students: 4988 (3480 with a failed topic not practised for > 14 days).

| Model | ndcg@10 | recall@20 | recall@5 | recall@10_old |
|---|---|---|---|---|
| Random (mean of seeds) | 0.5747 | 0.3804 | 0.0954 | 0.1864 |
| Most-popular (attempt frequency) | 0.8297 | 0.4680 | 0.1444 | 0.0771 |
| Global error rate | 0.5771 | 0.3808 | 0.0965 | 0.3006 |
| Personal error rate | 0.6145 | 0.3984 | 0.1026 | 0.2733 |
| Recent activity (answers in the last 100) | 0.8104 | 0.4533 | 0.1440 | 0.0214 |
| Full model (NLGCL + forgetting) | 0.8436 | 0.4877 | 0.1480 | 0.0757 |
| NLGCL, no forgetting | 0.8419 | 0.4864 | 0.1474 | 0.1107 |
| LightGCN + forgetting | 0.8419 | 0.4864 | 0.1475 | 0.0792 |
| LightGCN | 0.8405 | 0.4858 | 0.1471 | 0.1102 |

## Gaps (paired bootstrap over students, 95% interval)

| Model | Reference | Metric | Gap | 95% CI | Significant |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | Random (mean of seeds) | ndcg@10 | +0.2689 | [+0.2642, +0.2733] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@20 | +0.1074 | [+0.1052, +0.1096] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@10_old | -0.1107 | [-0.1161, -0.1052] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | ndcg@10 | +0.0139 | [+0.0119, +0.0160] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@20 | +0.0197 | [+0.0180, +0.0213] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@10_old | -0.0014 | [-0.0057, +0.0032] | no |
| Full model (NLGCL + forgetting) | Global error rate | ndcg@10 | +0.2665 | [+0.2590, +0.2727] | yes |
| Full model (NLGCL + forgetting) | Global error rate | recall@20 | +0.1069 | [+0.1036, +0.1102] | yes |
| Full model (NLGCL + forgetting) | Global error rate | recall@10_old | -0.2249 | [-0.2363, -0.2138] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | ndcg@10 | +0.2291 | [+0.2219, +0.2356] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | recall@20 | +0.0894 | [+0.0860, +0.0925] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | recall@10_old | -0.1976 | [-0.2087, -0.1862] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 100) | ndcg@10 | +0.0332 | [+0.0299, +0.0369] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 100) | recall@20 | +0.0344 | [+0.0322, +0.0367] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 100) | recall@10_old | +0.0543 | [+0.0495, +0.0590] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | ndcg@10 | +0.2672 | [+0.2623, +0.2716] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@20 | +0.1060 | [+0.1037, +0.1083] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@10_old | -0.0757 | [-0.0819, -0.0691] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.0122 | [+0.0100, +0.0144] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@20 | +0.0183 | [+0.0166, +0.0202] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@10_old | +0.0336 | [+0.0282, +0.0391] | yes |
| NLGCL, no forgetting | Global error rate | ndcg@10 | +0.2648 | [+0.2570, +0.2711] | yes |
| NLGCL, no forgetting | Global error rate | recall@20 | +0.1056 | [+0.1022, +0.1088] | yes |
| NLGCL, no forgetting | Global error rate | recall@10_old | -0.1899 | [-0.2012, -0.1776] | yes |
| NLGCL, no forgetting | Personal error rate | ndcg@10 | +0.2274 | [+0.2202, +0.2339] | yes |
| NLGCL, no forgetting | Personal error rate | recall@20 | +0.0880 | [+0.0846, +0.0912] | yes |
| NLGCL, no forgetting | Personal error rate | recall@10_old | -0.1627 | [-0.1734, -0.1512] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 100) | ndcg@10 | +0.0315 | [+0.0278, +0.0354] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 100) | recall@20 | +0.0331 | [+0.0307, +0.0353] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 100) | recall@10_old | +0.0893 | [+0.0833, +0.0959] | yes |
| LightGCN + forgetting | Random (mean of seeds) | ndcg@10 | +0.2672 | [+0.2624, +0.2715] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@20 | +0.1060 | [+0.1038, +0.1083] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@10_old | -0.1072 | [-0.1126, -0.1012] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.0122 | [+0.0100, +0.0143] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@20 | +0.0184 | [+0.0167, +0.0201] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@10_old | +0.0021 | [-0.0025, +0.0069] | no |
| LightGCN + forgetting | Global error rate | ndcg@10 | +0.2647 | [+0.2573, +0.2710] | yes |
| LightGCN + forgetting | Global error rate | recall@20 | +0.1056 | [+0.1023, +0.1088] | yes |
| LightGCN + forgetting | Global error rate | recall@10_old | -0.2214 | [-0.2328, -0.2097] | yes |
| LightGCN + forgetting | Personal error rate | ndcg@10 | +0.2274 | [+0.2198, +0.2337] | yes |
| LightGCN + forgetting | Personal error rate | recall@20 | +0.0880 | [+0.0847, +0.0913] | yes |
| LightGCN + forgetting | Personal error rate | recall@10_old | -0.1941 | [-0.2046, -0.1825] | yes |
| LightGCN + forgetting | Recent activity (answers in the last 100) | ndcg@10 | +0.0315 | [+0.0278, +0.0353] | yes |
| LightGCN + forgetting | Recent activity (answers in the last 100) | recall@20 | +0.0331 | [+0.0308, +0.0353] | yes |
| LightGCN + forgetting | Recent activity (answers in the last 100) | recall@10_old | +0.0578 | [+0.0527, +0.0631] | yes |
| LightGCN | Random (mean of seeds) | ndcg@10 | +0.2659 | [+0.2610, +0.2702] | yes |
| LightGCN | Random (mean of seeds) | recall@20 | +0.1054 | [+0.1032, +0.1076] | yes |
| LightGCN | Random (mean of seeds) | recall@10_old | -0.0762 | [-0.0824, -0.0698] | yes |
| LightGCN | Most-popular (attempt frequency) | ndcg@10 | +0.0108 | [+0.0086, +0.0130] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@20 | +0.0178 | [+0.0160, +0.0195] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@10_old | +0.0331 | [+0.0276, +0.0386] | yes |
| LightGCN | Global error rate | ndcg@10 | +0.2634 | [+0.2556, +0.2697] | yes |
| LightGCN | Global error rate | recall@20 | +0.1050 | [+0.1015, +0.1083] | yes |
| LightGCN | Global error rate | recall@10_old | -0.1904 | [-0.2020, -0.1783] | yes |
| LightGCN | Personal error rate | ndcg@10 | +0.2261 | [+0.2187, +0.2323] | yes |
| LightGCN | Personal error rate | recall@20 | +0.0874 | [+0.0840, +0.0907] | yes |
| LightGCN | Personal error rate | recall@10_old | -0.1632 | [-0.1738, -0.1515] | yes |
| LightGCN | Recent activity (answers in the last 100) | ndcg@10 | +0.0302 | [+0.0263, +0.0341] | yes |
| LightGCN | Recent activity (answers in the last 100) | recall@20 | +0.0325 | [+0.0302, +0.0348] | yes |
| LightGCN | Recent activity (answers in the last 100) | recall@10_old | +0.0888 | [+0.0830, +0.0951] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | ndcg@10 | +0.0017 | [+0.0004, +0.0030] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@20 | +0.0013 | [+0.0003, +0.0024] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@10_old | -0.0350 | [-0.0390, -0.0309] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | ndcg@10 | +0.0018 | [+0.0009, +0.0027] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@20 | +0.0013 | [+0.0007, +0.0019] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@10_old | -0.0035 | [-0.0060, -0.0014] | yes |
| Full model (NLGCL + forgetting) | LightGCN | ndcg@10 | +0.0031 | [+0.0017, +0.0045] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@20 | +0.0019 | [+0.0009, +0.0030] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@10_old | -0.0345 | [-0.0386, -0.0305] | yes |

## Sanity floor (Week 3 Q2)

Beats random and most-popular on NDCG@10 and Recall@20, with the 95% interval of every gap above 0.

- Full model (NLGCL + forgetting): **yes**
- NLGCL, no forgetting: **yes**
- LightGCN + forgetting: **yes**
- LightGCN: **yes**

## Learned forgetting parameters

beta > 0 means a topic gets riskier the longer it has not been practised.

| Model | beta | never-seen value | half-life median (days) | p10 | p90 |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | -0.1619 | +0.0977 | 5.39 | 4.65 | 11.88 |
| LightGCN + forgetting | -0.1589 | +0.0860 | 5.38 | 4.74 | 12.14 |
