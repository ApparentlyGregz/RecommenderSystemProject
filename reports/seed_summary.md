# Model variants over training seeds

Seeds [1, 2, 42]; mean ± standard deviation. Validation: best epoch (used for early stopping). Test: the kept checkpoint on the test cut.

| Variant | Seeds | Val NDCG@10 | Test NDCG@10 | Test Recall@5 |
|---|---|---|---|---|
| Full model (NLGCL + forgetting) | 3 | 0.7227 ± 0.0010 | 0.6856 ± 0.0010 | 0.4218 ± 0.0005 |
| NLGCL, no forgetting | 3 | 0.7101 ± 0.0027 | 0.6832 ± 0.0029 | 0.4173 ± 0.0013 |
| LightGCN + forgetting | 3 | 0.7222 ± 0.0005 | 0.6855 ± 0.0005 | 0.4227 ± 0.0024 |
| LightGCN | 3 | 0.7130 ± 0.0011 | 0.6849 ± 0.0010 | 0.4169 ± 0.0020 |

Test baselines (deterministic): Most-popular (attempt frequency) NDCG@10 0.5802, Recall@5 0.3599, Global error rate NDCG@10 0.6675, Recall@5 0.4471, Personal error rate NDCG@10 0.6617, Recall@5 0.4430
