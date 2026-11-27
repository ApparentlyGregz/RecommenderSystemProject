# Model variants over training seeds

Seeds [1, 2, 42]; mean ± standard deviation. Validation: best epoch (used for early stopping). Test: the kept checkpoint on the test cut.

| Variant | Seeds | Val NDCG@10 | Test NDCG@10 | Test Recall@5 |
|---|---|---|---|---|
| Full model (NLGCL + forgetting) | 3 | 0.6686 ± 0.0018 | 0.6508 ± 0.0018 | 0.3831 ± 0.0041 |
| NLGCL, no forgetting | 3 | 0.6689 ± 0.0045 | 0.6626 ± 0.0012 | 0.3931 ± 0.0113 |
| LightGCN + forgetting | 3 | 0.6690 ± 0.0024 | 0.6509 ± 0.0021 | 0.3851 ± 0.0015 |
| LightGCN | 3 | 0.6698 ± 0.0052 | 0.6631 ± 0.0016 | 0.3944 ± 0.0090 |

Test baselines (deterministic): Most-popular (attempt frequency) NDCG@10 0.5504, Recall@5 0.2833, Global error rate NDCG@10 0.6791, Recall@5 0.3860, Personal error rate NDCG@10 0.6528, Recall@5 0.3775
