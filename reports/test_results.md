# Results on the test cut

Commit `f2876a3`, 2026-11-27 09:58. Label: `above_mean`. Models: full, cl_noforget, nocl_forget, nocl_noforget. 1000 bootstrap resamples, seed 42.

Eligible students: 710 (97 with a failed topic not practised for > 14 days).

| Model | ndcg@10 | recall@20 | recall@5 | recall@10_old |
|---|---|---|---|---|
| Random (mean of seeds) | 0.5281 | 0.9936 | 0.3151 | 0.6629 |
| Most-popular (attempt frequency) | 0.5802 | 0.9935 | 0.3599 | 0.9412 |
| Global error rate | 0.6675 | 0.9951 | 0.4471 | 0.4483 |
| Personal error rate | 0.6617 | 0.9954 | 0.4430 | 0.5188 |
| Recent activity (answers in the last window) | 0.5975 | 0.9942 | 0.3540 | 0.4626 |
| Full model (NLGCL + forgetting) | 0.6867 | 0.9965 | 0.4217 | 0.6900 |
| NLGCL, no forgetting | 0.6858 | 0.9967 | 0.4192 | 0.7892 |
| LightGCN + forgetting | 0.6859 | 0.9959 | 0.4209 | 0.6681 |
| LightGCN | 0.6853 | 0.9966 | 0.4176 | 0.7855 |

## Gaps (paired bootstrap over students, 95% interval)

| Model | Reference | Metric | Gap | 95% CI | Significant |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | Random (mean of seeds) | ndcg@10 | +0.1586 | [+0.1462, +0.1716] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@20 | +0.0029 | [+0.0014, +0.0045] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@10_old | +0.0271 | [-0.0521, +0.0960] | no |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | ndcg@10 | +0.1065 | [+0.0882, +0.1243] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@20 | +0.0030 | [+0.0007, +0.0054] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@10_old | -0.2512 | [-0.3414, -0.1558] | yes |
| Full model (NLGCL + forgetting) | Global error rate | ndcg@10 | +0.0191 | [+0.0081, +0.0301] | yes |
| Full model (NLGCL + forgetting) | Global error rate | recall@20 | +0.0014 | [-0.0001, +0.0032] | no |
| Full model (NLGCL + forgetting) | Global error rate | recall@10_old | +0.2417 | [+0.1449, +0.3291] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | ndcg@10 | +0.0249 | [+0.0129, +0.0363] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | recall@20 | +0.0011 | [-0.0004, +0.0029] | no |
| Full model (NLGCL + forgetting) | Personal error rate | recall@10_old | +0.1712 | [+0.0676, +0.2681] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | ndcg@10 | +0.0892 | [+0.0726, +0.1062] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | recall@20 | +0.0024 | [+0.0006, +0.0045] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | recall@10_old | +0.2274 | [+0.1147, +0.3283] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | ndcg@10 | +0.1577 | [+0.1453, +0.1707] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@20 | +0.0031 | [+0.0015, +0.0048] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@10_old | +0.1262 | [+0.0535, +0.1981] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.1056 | [+0.0891, +0.1223] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@20 | +0.0032 | [+0.0008, +0.0056] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@10_old | -0.1521 | [-0.2304, -0.0750] | yes |
| NLGCL, no forgetting | Global error rate | ndcg@10 | +0.0182 | [+0.0075, +0.0293] | yes |
| NLGCL, no forgetting | Global error rate | recall@20 | +0.0016 | [-0.0000, +0.0034] | no |
| NLGCL, no forgetting | Global error rate | recall@10_old | +0.3409 | [+0.2479, +0.4299] | yes |
| NLGCL, no forgetting | Personal error rate | ndcg@10 | +0.0240 | [+0.0122, +0.0355] | yes |
| NLGCL, no forgetting | Personal error rate | recall@20 | +0.0013 | [-0.0003, +0.0030] | no |
| NLGCL, no forgetting | Personal error rate | recall@10_old | +0.2703 | [+0.1713, +0.3731] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | ndcg@10 | +0.0883 | [+0.0719, +0.1047] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | recall@20 | +0.0025 | [+0.0009, +0.0045] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | recall@10_old | +0.3265 | [+0.2184, +0.4324] | yes |
| LightGCN + forgetting | Random (mean of seeds) | ndcg@10 | +0.1578 | [+0.1453, +0.1706] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@20 | +0.0023 | [+0.0007, +0.0040] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@10_old | +0.0052 | [-0.0764, +0.0778] | no |
| LightGCN + forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.1057 | [+0.0878, +0.1234] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@20 | +0.0024 | [+0.0002, +0.0049] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@10_old | -0.2731 | [-0.3682, -0.1788] | yes |
| LightGCN + forgetting | Global error rate | ndcg@10 | +0.0183 | [+0.0073, +0.0296] | yes |
| LightGCN + forgetting | Global error rate | recall@20 | +0.0008 | [-0.0007, +0.0027] | no |
| LightGCN + forgetting | Global error rate | recall@10_old | +0.2198 | [+0.1308, +0.3073] | yes |
| LightGCN + forgetting | Personal error rate | ndcg@10 | +0.0242 | [+0.0122, +0.0361] | yes |
| LightGCN + forgetting | Personal error rate | recall@20 | +0.0005 | [-0.0009, +0.0022] | no |
| LightGCN + forgetting | Personal error rate | recall@10_old | +0.1493 | [+0.0413, +0.2501] | yes |
| LightGCN + forgetting | Recent activity (answers in the last window) | ndcg@10 | +0.0884 | [+0.0718, +0.1051] | yes |
| LightGCN + forgetting | Recent activity (answers in the last window) | recall@20 | +0.0018 | [+0.0000, +0.0040] | yes |
| LightGCN + forgetting | Recent activity (answers in the last window) | recall@10_old | +0.2055 | [+0.0900, +0.3080] | yes |
| LightGCN | Random (mean of seeds) | ndcg@10 | +0.1571 | [+0.1451, +0.1702] | yes |
| LightGCN | Random (mean of seeds) | recall@20 | +0.0030 | [+0.0013, +0.0046] | yes |
| LightGCN | Random (mean of seeds) | recall@10_old | +0.1225 | [+0.0486, +0.1930] | yes |
| LightGCN | Most-popular (attempt frequency) | ndcg@10 | +0.1051 | [+0.0886, +0.1214] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@20 | +0.0031 | [+0.0007, +0.0056] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@10_old | -0.1558 | [-0.2358, -0.0764] | yes |
| LightGCN | Global error rate | ndcg@10 | +0.0177 | [+0.0074, +0.0289] | yes |
| LightGCN | Global error rate | recall@20 | +0.0015 | [-0.0002, +0.0033] | no |
| LightGCN | Global error rate | recall@10_old | +0.3372 | [+0.2453, +0.4290] | yes |
| LightGCN | Personal error rate | ndcg@10 | +0.0235 | [+0.0117, +0.0347] | yes |
| LightGCN | Personal error rate | recall@20 | +0.0011 | [-0.0005, +0.0029] | no |
| LightGCN | Personal error rate | recall@10_old | +0.2667 | [+0.1661, +0.3696] | yes |
| LightGCN | Recent activity (answers in the last window) | ndcg@10 | +0.0878 | [+0.0719, +0.1037] | yes |
| LightGCN | Recent activity (answers in the last window) | recall@20 | +0.0024 | [+0.0007, +0.0044] | yes |
| LightGCN | Recent activity (answers in the last window) | recall@10_old | +0.3229 | [+0.2176, +0.4309] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | ndcg@10 | +0.0009 | [-0.0046, +0.0063] | no |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@20 | -0.0002 | [-0.0010, +0.0007] | no |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@10_old | -0.0992 | [-0.1550, -0.0505] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | ndcg@10 | +0.0008 | [-0.0008, +0.0027] | no |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@20 | +0.0006 | [+0.0000, +0.0013] | no |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@10_old | +0.0219 | [+0.0000, +0.0541] | no |
| Full model (NLGCL + forgetting) | LightGCN | ndcg@10 | +0.0014 | [-0.0039, +0.0068] | no |
| Full model (NLGCL + forgetting) | LightGCN | recall@20 | -0.0000 | [-0.0009, +0.0009] | no |
| Full model (NLGCL + forgetting) | LightGCN | recall@10_old | -0.0955 | [-0.1492, -0.0451] | yes |

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
| Full model (NLGCL + forgetting) | +0.0493 | +0.2042 | 6.93 | 5.27 | 8.63 |
| LightGCN + forgetting | +0.0399 | +0.1871 | 6.35 | 5.25 | 8.26 |
