# Table 3. Fair external baselines on the reconstructed dataset

| Property | Poly-Prophet recomputed | Mean predictor | Morgan-Ridge | Morgan-RF |
|---|---|---|---|---|
| Tg | 53.4416 ± 4.79267 | 88.1043 ± 1.18048 | 55.3644 ± 1.98369 | 81.4634 ± 1.0399 |
| FFV | 0.00506851 ± 0.000312352 | 0.0205689 ± 0.000264578 | 0.00990288 ± 0.00021127 | 0.0154917 ± 0.000343747 |
| Tc | 0.0310845 ± 0.00290648 | 0.0773953 ± 0.00392048 | 0.0381504 ± 0.00339242 | 0.0647182 ± 0.00377153 |
| Density | 0.0283528 ± 0.00375812 | 0.109455 ± 0.004762 | 0.0531116 ± 0.00562407 | 0.103837 ± 0.00427553 |
| Rg | 1.40207 ± 0.143844 | 3.96293 ± 0.0557593 | 1.66112 ± 0.224365 | 3.56111 ± 0.090577 |

All rows use the same reconstructed dataset, target-specific labels, grouped outer five-fold splits, inner training-only model selection, OOF MAE, and sample SD across folds.
Morgan fingerprints are RDKit radius=2, 2048 bits. Morgan-Ridge selects alpha on the inner split; Morgan-RF selects minimum leaf size on the inner split. Mean predictor uses only the outer-training mean.
External reported values from other papers are not inserted here because their datasets, targets, units, or metrics differ. See baseline_registry.json and baseline_registry.md.
