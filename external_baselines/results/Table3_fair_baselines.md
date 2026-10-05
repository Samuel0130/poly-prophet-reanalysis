# Table 3. Fair external baselines on the reconstructed dataset

| Property | Poly-Prophet recomputed | Mean predictor | Morgan-Ridge | Morgan-RF | Chen 2021 RNN (transferred) | Zhang 2024 PU-GN (adapted) |
|---|---|---|---|---|---|---|
| Tg | 53.4416 ± 4.79267 | 88.1043 ± 1.18048 | 55.3644 ± 1.98369 | 81.4634 ± 1.0399 | 52.6316 ± 2.63499 | 66.3929 ± 14.6393 |
| FFV | 0.00506851 ± 0.000312352 | 0.0205689 ± 0.000264578 | 0.00990288 ± 0.00021127 | 0.0154917 ± 0.000343747 | 0.00809984 ± 0.000308511 | 0.0122433 ± 0.000477035 |
| Tc | 0.0310845 ± 0.00290648 | 0.0773953 ± 0.00392048 | 0.0381504 ± 0.00339242 | 0.0647182 ± 0.00377153 | 0.0372754 ± 0.00688223 | 0.0441656 ± 0.00461336 |
| Density | 0.0283528 ± 0.00375812 | 0.109455 ± 0.004762 | 0.0531116 ± 0.00562407 | 0.103837 ± 0.00427553 | 0.0380184 ± 0.0109614 | 0.0679547 ± 0.0167819 |
| Rg | 1.40207 ± 0.143844 | 3.96293 ± 0.0557593 | 1.66112 ± 0.224365 | 3.56111 ± 0.090577 | 1.89917 ± 0.182123 | 2.21468 ± 0.178176 |

All rows use the same reconstructed dataset, target-specific labels, grouped outer five-fold splits, inner training-only model selection, OOF MAE, and sample SD across folds.
Morgan fingerprints are RDKit radius=2, 2048 bits. Morgan-Ridge selects alpha on the inner split; Morgan-RF selects minimum leaf size on the inner split. Mean predictor uses only the outer-training mean.
External reported values from other papers are not inserted here because their datasets, targets, units, or metrics differ. See baseline_registry.json and baseline_registry.md.

Chen uses the downloaded author getRNNmodel factory with the published best architecture (120 chars, embedding15, two BiLSTM60, TimeDistributedDense30); Keras3 torch backend, fold-local vocabulary and target scaling, inner MSE early stopping (patience10, maximum500 epochs), outer-train refit. The original paper is Tg-only; other targets are explicitly method transfers.
PU-GN uses the downloaded author-team SolubilityGN code/config (3 layers, hidden32, sum aggregation, dropout enabled, Adam .001, L1 .0001). It is an adaptation: pandas and scatter compatibility fixes, per-molecule unit encoding, and one whole-repeat-unit MACCS167 node for failed decompositions. It must not be described as exact reproduction of the original Polymer-Unit Graph paper.
PURS decomposition failed for 2236/8972 structures. Fallback by labeled target: Tg 202/557, FFV 1729/7892, Tc 463/866, Density 340/613, Rg 343/614. Every original test row is retained. Detailed conversion audit and per-fold fallback counts are saved.
Model-selection budgets and stopping criteria differ between methods and are documented; same data and outer folds alone do not establish exhaustive equal-budget optimization. RF currently searches min_samples_leaf=50 or200 only; this restricted RF is preliminary, not evidence of superiority over a well-tuned RF.
Only targets with all five completed outer folds are included as numeric results.
