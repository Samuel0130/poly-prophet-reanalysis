
# PURS-PUFp-PUGraph

<img width="886" height="580" alt="image" src="https://github.com/user-attachments/assets/90747922-0dba-4197-b3a6-b6fef30e9ca7" />

Unified Python package for:

- `PURS`: polymer-unit recognition from polymer SMILES
- `PUFp`: polymer-unit fingerprint generation and lightweight ML workflows
- `PUGraph`: polymer-unit graph construction and wrapped graph-learning workflows

The package is organized around one shared pipeline:

`polymer SMILES -> PURS core -> PUFp / PUGraph -> downstream tasks`

## Start Here

This release keeps one public example:

- `examples/opecm_standard_demo/`


Recommended command:

```bash
python scripts/run_opecm_osc_unified_example.py --output-dir output/opecm_standard_demo
```

Optional graph execution:

```bash
python scripts/run_opecm_osc_unified_example.py --output-dir output/opecm_standard_demo --execute-graphs
```

## Installation

Basic install:

```bash
pip install -e .
```

Full workflow install:

```bash
pip install -e .[all]
```

Conda environments are also provided in `environments/`.

## Minimal Repository Structure

```text
PURS-PUFp-PUGraph/
  src/purs/                    package source code
  examples/opecm_standard_demo single public example
  data/testing/                standard sample table
  docs/                        minimal package docs
  scripts/                     runnable helper scripts
  tests/                       internal regression checks
```

## Documentation

- `examples/README.md`
- `docs/package_architecture_zh.md`

## License

This project is released under the MIT License. See `LICENSE`.

## When using this database in your research PLEASE cite the papers:  

[1] Xiumin Liu, Xinyue Zhang, Yabei Wu, Xuehai Ju, Caichao Ye*, Wenqing Zhang*. OPECM: A comprehensive dataset for organic polymer energy conversion materials enabling data-driven interpretative analysis. Phys. Rev. Mater., 2025, 9, 125405.  
[2] Xinyue Zhang, Ye Sheng, Xiumin Liu, Jiong Yang, William A. Goddard III, Caichao Ye*, Wenqing Zhang*. Polymer-unit Graph: Advancing Interpretability in Graph Neural Network Machine Learning for Organic Polymer Semiconductor Materials. J. Chem. Theory Comput., 2024, 20(7), 2908-2920.  
[3] Xinyue Zhang, Genwang Wei, Ye Sheng, Wenjun Bai, Jiong Yang, Wenqing Zhang*, Caichao Ye*. Polymer-Unit Fingerprint (PUFp): An Accessible Expression of Polymer Organic Semiconductors for Machine Learning. ACS Appl. Mater. Interfaces, 2023, 15(17), 21537–21548.  
[4] Xiumin Liu, Xinyue Zhang, Ye Sheng, Zihe Zhang, Pan Xiong*, Xuehai Ju*, Junwu Zhu, Caichao Ye*. Advancing Organic Photovoltaic Materials by Machine Learning-Driven Design with Polymer-Unit Fingerprints. npj Comput. Mater., 2025, 11, 107.  
[5] Caichao Ye, Tao Feng, Weishu Liu*, Wenqing Zhang*. Functional Unit: A New Perspective on Materials Science Research Paradigms. Acc. Mater. Res., 2025, 6(8), 914-920.

