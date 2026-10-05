# `pu_mpnn`

Place the migrated `PU-MPNN` workflow here.

Recommended cleanup goals:

- separate reusable library code from experiment scripts
- move heavy environment requirements into `environment-graph.yml`
- keep package-facing preprocessing details close to the generated graph configs

Current unified entry points:

- preprocessing wrapper: `python -m purs.graph.pu_mpnn.prepare`
- legacy training code remains in `src/purs/graph/pu_mpnn/legacy/`

The preprocessing wrapper now accepts explicit arguments:

```bash
python -m purs.graph.pu_mpnn.prepare \
  --input-csv output/graph/graph_input.csv \
  --output-pkl output/graph/pu_mpnn/genwl3.pkl
```

When `purs graph train --config ...` targets `backend: pu_mpnn`, the returned command now includes
the config-driven `input_csv`, `output_pkl`, `release_check.csv`, and SDF paths so the preprocessing
step is reproducible from the generated YAML alone.
