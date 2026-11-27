# Results on the test cut

Commit `c6a4f28`, 2026-11-27 09:12. Models: full, cl_noforget, nocl_forget, nocl_noforget. 1000 bootstrap resamples, seed 42.

Eligible students: 726 (131 with a failed topic not practised for > 7 days).

| Model | ndcg@10 | recall@20 | recall@5 | recall@10_old |
|---|---|---|---|---|
| Random (mean of seeds) | 0.4950 | 0.9541 | 0.2657 | 0.5498 |
| Most-popular (attempt frequency) | 0.5504 | 0.9680 | 0.2833 | 0.9159 |
| Global error rate | 0.6791 | 0.9809 | 0.3860 | 0.3928 |
| Personal error rate | 0.6528 | 0.9779 | 0.3775 | 0.4371 |
| Recent activity (answers in the last 30) | 0.6056 | 0.9803 | 0.3187 | 0.4148 |
| Full model (NLGCL + forgetting) | 0.6524 | 0.9833 | 0.3859 | 0.3842 |
| NLGCL, no forgetting | 0.6614 | 0.9850 | 0.4027 | 0.6100 |
| LightGCN + forgetting | 0.6536 | 0.9816 | 0.3850 | 0.4062 |
| LightGCN | 0.6624 | 0.9850 | 0.4020 | 0.6348 |

## Gaps (paired bootstrap over students, 95% interval)

| Model | Reference | Metric | Gap | 95% CI | Significant |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | Random (mean of seeds) | ndcg@10 | +0.1575 | [+0.1454, +0.1684] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@20 | +0.0292 | [+0.0248, +0.0334] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@10_old | -0.1656 | [-0.2230, -0.1045] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | ndcg@10 | +0.1020 | [+0.0857, +0.1191] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@20 | +0.0153 | [+0.0099, +0.0206] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@10_old | -0.5317 | [-0.6028, -0.4547] | yes |
| Full model (NLGCL + forgetting) | Global error rate | ndcg@10 | -0.0267 | [-0.0377, -0.0174] | yes |
| Full model (NLGCL + forgetting) | Global error rate | recall@20 | +0.0024 | [-0.0001, +0.0050] | no |
| Full model (NLGCL + forgetting) | Global error rate | recall@10_old | -0.0086 | [-0.0746, +0.0575] | no |
| Full model (NLGCL + forgetting) | Personal error rate | ndcg@10 | -0.0003 | [-0.0114, +0.0104] | no |
| Full model (NLGCL + forgetting) | Personal error rate | recall@20 | +0.0054 | [+0.0022, +0.0085] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | recall@10_old | -0.0529 | [-0.1223, +0.0185] | no |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 30) | ndcg@10 | +0.0468 | [+0.0313, +0.0616] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 30) | recall@20 | +0.0030 | [+0.0004, +0.0058] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last 30) | recall@10_old | -0.0306 | [-0.1016, +0.0459] | no |
| NLGCL, no forgetting | Random (mean of seeds) | ndcg@10 | +0.1664 | [+0.1539, +0.1784] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@20 | +0.0309 | [+0.0269, +0.0351] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@10_old | +0.0602 | [-0.0072, +0.1262] | no |
| NLGCL, no forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.1109 | [+0.0965, +0.1264] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@20 | +0.0170 | [+0.0119, +0.0222] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@10_old | -0.3060 | [-0.3783, -0.2283] | yes |
| NLGCL, no forgetting | Global error rate | ndcg@10 | -0.0178 | [-0.0289, -0.0071] | yes |
| NLGCL, no forgetting | Global error rate | recall@20 | +0.0041 | [+0.0017, +0.0068] | yes |
| NLGCL, no forgetting | Global error rate | recall@10_old | +0.2171 | [+0.1304, +0.3026] | yes |
| NLGCL, no forgetting | Personal error rate | ndcg@10 | +0.0086 | [-0.0030, +0.0196] | no |
| NLGCL, no forgetting | Personal error rate | recall@20 | +0.0071 | [+0.0040, +0.0101] | yes |
| NLGCL, no forgetting | Personal error rate | recall@10_old | +0.1728 | [+0.0835, +0.2657] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 30) | ndcg@10 | +0.0558 | [+0.0419, +0.0702] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 30) | recall@20 | +0.0047 | [+0.0024, +0.0074] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last 30) | recall@10_old | +0.1951 | [+0.1056, +0.2887] | yes |
| LightGCN + forgetting | Random (mean of seeds) | ndcg@10 | +0.1586 | [+0.1463, +0.1699] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@20 | +0.0275 | [+0.0231, +0.0317] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@10_old | -0.1436 | [-0.2034, -0.0761] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.1032 | [+0.0872, +0.1202] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@20 | +0.0136 | [+0.0081, +0.0190] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@10_old | -0.5097 | [-0.5849, -0.4278] | yes |
| LightGCN + forgetting | Global error rate | ndcg@10 | -0.0255 | [-0.0361, -0.0157] | yes |
| LightGCN + forgetting | Global error rate | recall@20 | +0.0007 | [-0.0020, +0.0034] | no |
| LightGCN + forgetting | Global error rate | recall@10_old | +0.0134 | [-0.0558, +0.0812] | no |
| LightGCN + forgetting | Personal error rate | ndcg@10 | +0.0009 | [-0.0103, +0.0119] | no |
| LightGCN + forgetting | Personal error rate | recall@20 | +0.0037 | [+0.0002, +0.0069] | yes |
| LightGCN + forgetting | Personal error rate | recall@10_old | -0.0309 | [-0.1045, +0.0483] | no |
| LightGCN + forgetting | Recent activity (answers in the last 30) | ndcg@10 | +0.0480 | [+0.0330, +0.0628] | yes |
| LightGCN + forgetting | Recent activity (answers in the last 30) | recall@20 | +0.0013 | [-0.0015, +0.0044] | no |
| LightGCN + forgetting | Recent activity (answers in the last 30) | recall@10_old | -0.0086 | [-0.0855, +0.0702] | no |
| LightGCN | Random (mean of seeds) | ndcg@10 | +0.1674 | [+0.1545, +0.1793] | yes |
| LightGCN | Random (mean of seeds) | recall@20 | +0.0310 | [+0.0270, +0.0353] | yes |
| LightGCN | Random (mean of seeds) | recall@10_old | +0.0850 | [+0.0200, +0.1530] | yes |
| LightGCN | Most-popular (attempt frequency) | ndcg@10 | +0.1119 | [+0.0977, +0.1273] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@20 | +0.0171 | [+0.0119, +0.0222] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@10_old | -0.2812 | [-0.3552, -0.2053] | yes |
| LightGCN | Global error rate | ndcg@10 | -0.0168 | [-0.0281, -0.0062] | yes |
| LightGCN | Global error rate | recall@20 | +0.0041 | [+0.0019, +0.0068] | yes |
| LightGCN | Global error rate | recall@10_old | +0.2419 | [+0.1542, +0.3295] | yes |
| LightGCN | Personal error rate | ndcg@10 | +0.0096 | [-0.0022, +0.0206] | no |
| LightGCN | Personal error rate | recall@20 | +0.0071 | [+0.0041, +0.0101] | yes |
| LightGCN | Personal error rate | recall@10_old | +0.1976 | [+0.1078, +0.2950] | yes |
| LightGCN | Recent activity (answers in the last 30) | ndcg@10 | +0.0568 | [+0.0434, +0.0708] | yes |
| LightGCN | Recent activity (answers in the last 30) | recall@20 | +0.0047 | [+0.0024, +0.0076] | yes |
| LightGCN | Recent activity (answers in the last 30) | recall@10_old | +0.2200 | [+0.1318, +0.3154] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | ndcg@10 | -0.0089 | [-0.0157, -0.0026] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@20 | -0.0017 | [-0.0037, +0.0001] | no |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@10_old | -0.2257 | [-0.2890, -0.1679] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | ndcg@10 | -0.0012 | [-0.0043, +0.0019] | no |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@20 | +0.0017 | [+0.0004, +0.0032] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@10_old | -0.0220 | [-0.0562, +0.0051] | no |
| Full model (NLGCL + forgetting) | LightGCN | ndcg@10 | -0.0099 | [-0.0166, -0.0041] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@20 | -0.0018 | [-0.0037, -0.0001] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@10_old | -0.2505 | [-0.3174, -0.1870] | yes |

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
| Full model (NLGCL + forgetting) | -0.0983 | +0.1904 | 6.32 | 4.89 | 8.13 |
| LightGCN + forgetting | -0.0924 | +0.1709 | 5.50 | 4.66 | 7.95 |
