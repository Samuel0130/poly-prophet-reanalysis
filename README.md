# Poly-Prophet: audited re-analysis and external baselines

This repository contains the reproducible re-analysis of the Poly-Prophet polymer-property prediction manuscript.

The work is organized around the reviewer-requested tables:

- `table1_rebuilt/`: audited dataset composition, canonical-structure provenance, missingness, ranges, means and sample SD.
- `training/results/`: newly trained grouped five-fold OOF results and the seven-variant ablation (`Table4_ablation.docx`, `retraining_results.xlsx`).
- `external_baselines/`: baseline registry, literature comparability audit, and fair same-data Morgan/mean baselines (`results/Table3_fair_baselines.md`).
- `training/artifacts/`: lightweight dataset, folds and verification metadata. Large model/input caches are deliberately ignored and can be regenerated.
- `cnnlstm-bamod4c-molfold-aug-infer.ipynb`: user-specified original inference notebook.

## Reproduce

Use Python 3.12 and install `training/requirements.txt` in an isolated environment. Kaggle inputs are downloaded with `download_kaggle_inputs.py`; credentials are read by the official Kaggle client and are never committed.

```bash
.venv/bin/python download_kaggle_inputs.py
.venv/bin/python rebuild_table1.py
.venv/bin/python training/prepare.py
.venv/bin/python training/launch_experiment.py
.venv/bin/python external_baselines/run_classical.py
.venv/bin/python external_baselines/make_report.py
```

The primary results are a re-analysis under a documented grouped five-fold protocol, not a claim of exact numerical reproduction of the original manuscript. External literature values are kept separate when datasets, targets, units or metrics differ.
