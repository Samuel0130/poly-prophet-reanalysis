# Actual retraining and ablation results

| Model Variant | Tg | FFV | Tc | Density | Rg |
| --- | --- | --- | --- | --- | --- |
| Full Model | 53.44 ± 4.79 | 0.00507 ± 0.00031 | 0.03108 ± 0.00291 | 0.02835 ± 0.00376 | 1.402 ± 0.144 |
| w/o MolFormer | 56.14 ± 7.28 | 0.00490 ± 0.00029 | 0.03404 ± 0.00559 | 0.02769 ± 0.00414 | 1.507 ± 0.118 |
| w/o Branch Descriptors | 55.76 ± 4.17 | 0.00486 ± 0.00018 | 0.03032 ± 0.00557 | 0.02598 ± 0.00440 | 1.624 ± 0.128 |
| w/o CNN | 52.96 ± 3.66 | 0.00472 ± 0.00009 | 0.02953 ± 0.00415 | 0.02696 ± 0.00467 | 1.518 ± 0.205 |
| w/o LSTM | 54.25 ± 7.17 | 0.00512 ± 0.00014 | 0.03299 ± 0.00268 | 0.03190 ± 0.00662 | 1.649 ± 0.411 |
| w/o Augmentation | 55.52 ± 3.58 | 0.00493 ± 0.00025 | 0.03244 ± 0.00491 | 0.03179 ± 0.00538 | 1.567 ± 0.256 |
| Only SMILES Sequence | 56.33 ± 4.36 | 0.00495 ± 0.00023 | 0.03123 ± 0.00633 | 0.03106 ± 0.00466 | 1.741 ± 0.190 |

Values are outer five-fold MAE mean ± sample SD, in original units. Each filled cell represents five completed, newly trained models. Pending cells are not results.
Variants share outer folds and inner early-stopping rows. Selection uses a training-only inner holdout; models are refit on all outer-training rows for the selected number of epochs. Outer validation never selects epochs.
See performance_summary.csv for OOF MAE, RMSE, R² and 95% group bootstrap confidence intervals. Ablation p-values use paired five-fold MAE differences with corrected resampled variance (1/5 + mean n_test/n_train) and Holm correction across available comparisons. With only five folds, inferential conclusions remain limited.
Architecture is the user-specified original inference CNN/BiLSTM, using its fixed 64-dimensional MolFormer extraction. Batch size, fold-local preprocessing, grouping and nested early stopping were updated and documented in experiment_config.json. These are new experiments, not exact numerical reproduction of the paper.
The original descriptor function declares 17 fields, but only six are nonzero in these data. The descriptor ablation measures the contribution of those six active original fields; see artifacts/descriptor_audit.csv.
Grouping combines canonical non-isomeric SMILES; general polymer repeat-unit equivalence is not claimed. Supplemental labels use the author training data policy from Table 1. Frozen MolFormer features have no descriptor fallback. Only training SMILES are augmented.
