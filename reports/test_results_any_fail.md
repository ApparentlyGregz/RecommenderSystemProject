# Results on the test cut

Commit `feba0a8 + uncommitted changes in src/`, 2026-11-27 09:54. Label: `any_fail`. Models: full, cl_noforget, nocl_forget, nocl_noforget. 1000 bootstrap resamples, seed 42.

Eligible students: 703 (102 with a failed topic not practised for > 14 days).

| Model | ndcg@10 | recall@20 | recall@5 | recall@10_old |
|---|---|---|---|---|
| Random (mean of seeds) | 0.6014 | 0.9935 | 0.3129 | 0.6421 |
| Most-popular (attempt frequency) | 0.7490 | 0.9941 | 0.4143 | 0.9586 |
| Global error rate | 0.7229 | 0.9952 | 0.4124 | 0.4467 |
| Personal error rate | 0.7164 | 0.9955 | 0.4083 | 0.5259 |
| Recent activity (answers in the last window) | 0.6542 | 0.9943 | 0.3213 | 0.4556 |
| Full model (NLGCL + forgetting) | 0.7806 | 0.9963 | 0.4149 | 0.6685 |
| NLGCL, no forgetting | 0.7776 | 0.9967 | 0.4087 | 0.7450 |
| LightGCN + forgetting | 0.7801 | 0.9959 | 0.4137 | 0.6527 |
| LightGCN | 0.7750 | 0.9966 | 0.4061 | 0.7428 |

## Gaps (paired bootstrap over students, 95% interval)

| Model | Reference | Metric | Gap | 95% CI | Significant |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | Random (mean of seeds) | ndcg@10 | +0.1791 | [+0.1690, +0.1896] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@20 | +0.0028 | [+0.0013, +0.0044] | yes |
| Full model (NLGCL + forgetting) | Random (mean of seeds) | recall@10_old | +0.0264 | [-0.0418, +0.0989] | no |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | ndcg@10 | +0.0316 | [+0.0182, +0.0455] | yes |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@20 | +0.0022 | [-0.0001, +0.0048] | no |
| Full model (NLGCL + forgetting) | Most-popular (attempt frequency) | recall@10_old | -0.2901 | [-0.3735, -0.2030] | yes |
| Full model (NLGCL + forgetting) | Global error rate | ndcg@10 | +0.0576 | [+0.0487, +0.0667] | yes |
| Full model (NLGCL + forgetting) | Global error rate | recall@20 | +0.0011 | [-0.0003, +0.0029] | no |
| Full model (NLGCL + forgetting) | Global error rate | recall@10_old | +0.2217 | [+0.1404, +0.3061] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | ndcg@10 | +0.0642 | [+0.0550, +0.0739] | yes |
| Full model (NLGCL + forgetting) | Personal error rate | recall@20 | +0.0008 | [-0.0006, +0.0027] | no |
| Full model (NLGCL + forgetting) | Personal error rate | recall@10_old | +0.1426 | [+0.0484, +0.2424] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | ndcg@10 | +0.1263 | [+0.1121, +0.1406] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | recall@20 | +0.0020 | [+0.0005, +0.0038] | yes |
| Full model (NLGCL + forgetting) | Recent activity (answers in the last window) | recall@10_old | +0.2129 | [+0.1106, +0.3098] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | ndcg@10 | +0.1762 | [+0.1661, +0.1874] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@20 | +0.0032 | [+0.0015, +0.0049] | yes |
| NLGCL, no forgetting | Random (mean of seeds) | recall@10_old | +0.1029 | [+0.0324, +0.1697] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.0286 | [+0.0154, +0.0418] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@20 | +0.0026 | [+0.0001, +0.0051] | yes |
| NLGCL, no forgetting | Most-popular (attempt frequency) | recall@10_old | -0.2136 | [-0.2918, -0.1383] | yes |
| NLGCL, no forgetting | Global error rate | ndcg@10 | +0.0547 | [+0.0447, +0.0635] | yes |
| NLGCL, no forgetting | Global error rate | recall@20 | +0.0015 | [-0.0001, +0.0034] | no |
| NLGCL, no forgetting | Global error rate | recall@10_old | +0.2982 | [+0.2222, +0.3861] | yes |
| NLGCL, no forgetting | Personal error rate | ndcg@10 | +0.0612 | [+0.0514, +0.0709] | yes |
| NLGCL, no forgetting | Personal error rate | recall@20 | +0.0012 | [-0.0004, +0.0030] | no |
| NLGCL, no forgetting | Personal error rate | recall@10_old | +0.2190 | [+0.1230, +0.3165] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | ndcg@10 | +0.1234 | [+0.1100, +0.1363] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | recall@20 | +0.0024 | [+0.0008, +0.0042] | yes |
| NLGCL, no forgetting | Recent activity (answers in the last window) | recall@10_old | +0.2893 | [+0.1885, +0.3901] | yes |
| LightGCN + forgetting | Random (mean of seeds) | ndcg@10 | +0.1786 | [+0.1687, +0.1894] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@20 | +0.0024 | [+0.0009, +0.0039] | yes |
| LightGCN + forgetting | Random (mean of seeds) | recall@10_old | +0.0106 | [-0.0620, +0.0864] | no |
| LightGCN + forgetting | Most-popular (attempt frequency) | ndcg@10 | +0.0311 | [+0.0173, +0.0451] | yes |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@20 | +0.0018 | [-0.0004, +0.0042] | no |
| LightGCN + forgetting | Most-popular (attempt frequency) | recall@10_old | -0.3059 | [-0.3894, -0.2179] | yes |
| LightGCN + forgetting | Global error rate | ndcg@10 | +0.0571 | [+0.0483, +0.0659] | yes |
| LightGCN + forgetting | Global error rate | recall@20 | +0.0006 | [-0.0007, +0.0024] | no |
| LightGCN + forgetting | Global error rate | recall@10_old | +0.2059 | [+0.1283, +0.2921] | yes |
| LightGCN + forgetting | Personal error rate | ndcg@10 | +0.0637 | [+0.0546, +0.0734] | yes |
| LightGCN + forgetting | Personal error rate | recall@20 | +0.0004 | [-0.0010, +0.0022] | no |
| LightGCN + forgetting | Personal error rate | recall@10_old | +0.1268 | [+0.0277, +0.2300] | yes |
| LightGCN + forgetting | Recent activity (answers in the last window) | ndcg@10 | +0.1258 | [+0.1116, +0.1401] | yes |
| LightGCN + forgetting | Recent activity (answers in the last window) | recall@20 | +0.0015 | [-0.0000, +0.0033] | no |
| LightGCN + forgetting | Recent activity (answers in the last window) | recall@10_old | +0.1971 | [+0.0950, +0.2958] | yes |
| LightGCN | Random (mean of seeds) | ndcg@10 | +0.1736 | [+0.1634, +0.1844] | yes |
| LightGCN | Random (mean of seeds) | recall@20 | +0.0031 | [+0.0014, +0.0048] | yes |
| LightGCN | Random (mean of seeds) | recall@10_old | +0.1007 | [+0.0317, +0.1683] | yes |
| LightGCN | Most-popular (attempt frequency) | ndcg@10 | +0.0260 | [+0.0129, +0.0394] | yes |
| LightGCN | Most-popular (attempt frequency) | recall@20 | +0.0025 | [-0.0000, +0.0050] | no |
| LightGCN | Most-popular (attempt frequency) | recall@10_old | -0.2158 | [-0.2967, -0.1388] | yes |
| LightGCN | Global error rate | ndcg@10 | +0.0521 | [+0.0424, +0.0610] | yes |
| LightGCN | Global error rate | recall@20 | +0.0014 | [-0.0003, +0.0033] | no |
| LightGCN | Global error rate | recall@10_old | +0.2960 | [+0.2208, +0.3836] | yes |
| LightGCN | Personal error rate | ndcg@10 | +0.0586 | [+0.0486, +0.0684] | yes |
| LightGCN | Personal error rate | recall@20 | +0.0011 | [-0.0004, +0.0030] | no |
| LightGCN | Personal error rate | recall@10_old | +0.2169 | [+0.1219, +0.3137] | yes |
| LightGCN | Recent activity (answers in the last window) | ndcg@10 | +0.1208 | [+0.1074, +0.1338] | yes |
| LightGCN | Recent activity (answers in the last window) | recall@20 | +0.0023 | [+0.0007, +0.0041] | yes |
| LightGCN | Recent activity (answers in the last window) | recall@10_old | +0.2872 | [+0.1889, +0.3891] | yes |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | ndcg@10 | +0.0030 | [-0.0017, +0.0074] | no |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@20 | -0.0004 | [-0.0012, +0.0004] | no |
| Full model (NLGCL + forgetting) | NLGCL, no forgetting | recall@10_old | -0.0765 | [-0.1213, -0.0350] | yes |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | ndcg@10 | +0.0005 | [-0.0011, +0.0020] | no |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@20 | +0.0004 | [+0.0000, +0.0010] | no |
| Full model (NLGCL + forgetting) | LightGCN + forgetting | recall@10_old | +0.0158 | [+0.0000, +0.0403] | no |
| Full model (NLGCL + forgetting) | LightGCN | ndcg@10 | +0.0055 | [+0.0011, +0.0100] | yes |
| Full model (NLGCL + forgetting) | LightGCN | recall@20 | -0.0003 | [-0.0011, +0.0005] | no |
| Full model (NLGCL + forgetting) | LightGCN | recall@10_old | -0.0743 | [-0.1210, -0.0329] | yes |

## Sanity floor (Week 3 Q2)

Beats random and most-popular on NDCG@10 and Recall@20, with the 95% interval of every gap above 0.

- Full model (NLGCL + forgetting): **no**
- NLGCL, no forgetting: **yes**
- LightGCN + forgetting: **no**
- LightGCN: **no**

## Learned forgetting parameters

beta > 0 means a topic gets riskier the longer it has not been practised.

| Model | beta | never-seen value | half-life median (days) | p10 | p90 |
|---|---|---|---|---|---|
| Full model (NLGCL + forgetting) | +0.0493 | +0.2042 | 6.93 | 5.27 | 8.63 |
| LightGCN + forgetting | +0.0399 | +0.1871 | 6.35 | 5.25 | 8.26 |
