# Results on the test cut

Commit `c6a4f28`, 2026-11-27 09:10. Models: full, cl_noforget, nocl_forget, nocl_noforget. 1000 bootstrap resamples, seed 42.

Eligible students: 726 (131 with a failed topic not practised for > 7 days).

| Model | ndcg@10 | recall@20 | recall@5 | recall@10_old |
|---|---|---|---|---|
| Random (mean of seeds) | 0.4950 | 0.9541 | 0.2657 | 0.5498 |
| Most-popular (attempt frequency) | 0.5504 | 0.9680 | 0.2833 | 0.9159 |
| Global error rate | 0.6791 | 0.9809 | 0.3860 | 0.3928 |
| Personal error rate | 0.6528 | 0.9779 | 0.3775 | 0.4371 |
| Recent activity (answers in the last 30) | 0.6056 | 0.9803 | 0.3187 | 0.4148 |
| Full model (NLGCL + forgetting) | 0.6475 | 0.9819 | 0.3790 | 0.3682 |
| NLGCL, no forgetting | 0.6603 | 0.9850 | 0.4011 | 0.6789 |
| LightGCN + forgetting | 0.6459 | 0.9805 | 0.3818 | 0.4006 |
| LightGCN | 0.6590 | 0.9851 | 0.3999 | 0.6765 |

## Gaps (paired bootstrap over students, 95% interval)

| Model | Reference | Metric | Gap | 95% CI | Significant |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | Random (mean of seeds) | ndcg@10 | +0.1526 | [+0.1400, +0.1639] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@20 | +0.0278 | [+0.0234, +0.0322] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@10_old | -0.1816 | [-0.2421, -0.1173] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | ndcg@10 | +0.0971 | [+0.0817, +0.1136] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@20 | +0.0139 | [+0.0086, +0.0195] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@10_old | -0.5477 | [-0.6206, -0.4672] | yes |
| Full model (NLGCL + forgetting) | Global error rate | ndcg@10 | -0.0316 | [-0.0425, -0.0215] | yes |
| Full model (NLGCL + forgetting) | Global error rate | recall@20 | +0.0010 | [-0.0018, +0.0040] | no |
| Full model (NLGCL + forgetting) | Global error rate | recall@10_old | -0.0246 | [-0.0927, +0.0384] | no |
| Full model (NLGCL + forgetting) | Personal error rate | ndcg@10 | -0.0052 | [-0.0172, +0.0058] | no |
| Full model (NLGCL + forgetting) | Personal error rate | recall@20 | +0.0040 | [+0.0007, +0.0073] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | recall@10_old | -0.0689 | [-0.1402, +0.0022] | no |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 30) | ndcg@10 | +0.0419 | [+0.0262, +0.0572] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 30) | recall@20 | +0.0016 | [-0.0014, +0.0048] | no |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 30) | recall@10_old | -0.0466 | [-0.1170, +0.0272] | no |
| NLGCL, no forgetting | Random (mean of seeds) | ndcg@10 | +0.1653 | [+0.1531, +0.1769] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@20 | +0.0310 | [+0.0270, +0.0353] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@10_old | +0.1291 | [+0.0612, +0.1924] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.1099 | [+0.0959, +0.1249] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@20 | +0.0171 | [+0.0119, +0.0223] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@10_old | -0.2370 | [-0.3122, -0.1645] | yes |
| NLGCL, no forgetting | Global error rate | ndcg@10 | -0.0188 | [-0.0298, -0.0080] | yes |
| NLGCL, no forgetting | Global error rate | recall@20 | +0.0041 | [+0.0018, +0.0069] | yes |
| NLGCL, no forgetting | Global error rate | recall@10_old | +0.2861 | [+0.2000, +0.3727] | yes |
| NLGCL, no forgetting | Personal error rate | ndcg@10 | +0.0076 | [-0.0042, +0.0194] | no |
| NLGCL, no forgetting | Personal error rate | recall@20 | +0.0071 | [+0.0041, +0.0103] | yes |
| NLGCL, no forgetting | Personal error rate | recall@10_old | +0.2418 | [+0.1521, +0.3332] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 30) | ndcg@10 | +0.0547 | [+0.0413, +0.0691] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 30) | recall@20 | +0.0047 | [+0.0024, +0.0077] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 30) | recall@10_old | +0.2641 | [+0.1783, +0.3566] | yes |
| LightGCN + forgetting | Random (mean of seeds) | ndcg@10 | +0.1510 | [+0.1384, +0.1625] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@20 | +0.0265 | [+0.0219, +0.0310] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@10_old | -0.1492 | [-0.2116, -0.0800] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.0955 | [+0.0799, +0.1123] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@20 | +0.0126 | [+0.0071, +0.0183] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@10_old | -0.5154 | [-0.5930, -0.4304] | yes |
| LightGCN + forgetting | Global error rate | ndcg@10 | -0.0332 | [-0.0443, -0.0229] | yes |
| LightGCN + forgetting | Global error rate | recall@20 | -0.0004 | [-0.0034, +0.0028] | no |
| LightGCN + forgetting | Global error rate | recall@10_old | +0.0077 | [-0.0586, +0.0749] | no |
| LightGCN + forgetting | Personal error rate | ndcg@10 | -0.0068 | [-0.0187, +0.0046] | no |
| LightGCN + forgetting | Personal error rate | recall@20 | +0.0027 | [-0.0010, +0.0059] | no |
| LightGCN + forgetting | Personal error rate | recall@10_old | -0.0366 | [-0.1077, +0.0366] | no |
| LightGCN + forgetting | Recent activity (answers in the last 30) | ndcg@10 | +0.0403 | [+0.0246, +0.0559] | yes |
| LightGCN + forgetting | Recent activity (answers in the last 30) | recall@20 | +0.0002 | [-0.0029, +0.0036] | no |
| LightGCN + forgetting | Recent activity (answers in the last 30) | recall@10_old | -0.0142 | [-0.0894, +0.0632] | no |
| LightGCN | Random (mean of seeds) | ndcg@10 | +0.1641 | [+0.1516, +0.1760] | yes |
| LightGCN | Random (mean of seeds) | recall@20 | +0.0311 | [+0.0270, +0.0352] | yes |
| LightGCN | Random (mean of seeds) | recall@10_old | +0.1267 | [+0.0628, +0.1902] | yes |
| LightGCN | Most-popular (attempt frequency) | ndcg@10 | +0.1086 | [+0.0945, +0.1234] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@20 | +0.0172 | [+0.0121, +0.0223] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@10_old | -0.2394 | [-0.3108, -0.1700] | yes |
| LightGCN | Global error rate | ndcg@10 | -0.0201 | [-0.0313, -0.0093] | yes |
| LightGCN | Global error rate | recall@20 | +0.0042 | [+0.0018, +0.0071] | yes |
| LightGCN | Global error rate | recall@10_old | +0.2837 | [+0.1958, +0.3700] | yes |
| LightGCN | Personal error rate | ndcg@10 | +0.0063 | [-0.0061, +0.0181] | no |
| LightGCN | Personal error rate | recall@20 | +0.0072 | [+0.0039, +0.0103] | yes |
| LightGCN | Personal error rate | recall@10_old | +0.2394 | [+0.1481, +0.3331] | yes |
| LightGCN | Recent activity (answers in the last 30) | ndcg@10 | +0.0534 | [+0.0400, +0.0680] | yes |
| LightGCN | Recent activity (answers in the last 30) | recall@20 | +0.0048 | [+0.0023, +0.0078] | yes |
| LightGCN | Recent activity (answers in the last 30) | recall@10_old | +0.2617 | [+0.1771, +0.3533] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | ndcg@10 | -0.0128 | [-0.0208, -0.0063] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@20 | -0.0031 | [-0.0052, -0.0012] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@10_old | -0.3107 | [-0.3809, -0.2436] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | ndcg@10 | +0.0016 | [-0.0022, +0.0054] | no |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@20 | +0.0014 | [+0.0001, +0.0027] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@10_old | -0.0324 | [-0.0638, -0.0047] | yes |
| Full model (NLGCL + forgetting) | LightGCN | ndcg@10 | -0.0115 | [-0.0190, -0.0051] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@20 | -0.0032 | [-0.0055, -0.0012] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@10_old | -0.3083 | [-0.3808, -0.2385] | yes |

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
| Full model (NLGCL + forgetting) | -0.1123 | +0.2151 | 6.35 | 4.70 | 8.42 |
| LightGCN + forgetting | -0.1069 | +0.2018 | 5.48 | 4.54 | 8.10 |
