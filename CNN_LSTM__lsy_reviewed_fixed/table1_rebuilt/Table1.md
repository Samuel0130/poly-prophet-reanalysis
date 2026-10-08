# Table 1. Composition and target distributions of the reconstructed polymer analysis dataset

**Unique canonical polymers: 8,972.**

| Property | Unit | Labeled polymers | Range | Mean ± SD | Missing, n (%) |
| --- | --- | --- | --- | --- | --- |
| Tg | °C | 557 | -148.03–472.25 | 99.69 ± 111.08 | 8,415 (93.79%) |
| FFV | dimensionless | 7892 | 0.22699–0.77710 | 0.36701 ± 0.02907 | 1,080 (12.04%) |
| Tc | W/(m·K) | 866 | 0.04650–1.59000 | 0.25664 ± 0.10128 | 8,106 (90.35%) |
| Density | g/cm³ | 613 | 0.74869–1.84100 | 0.98548 ± 0.14619 | 8,359 (93.17%) |
| Rg | Å | 614 | 9.728–34.673 | 16.420 ± 4.609 | 8,358 (93.16%) |

Notes:
1. The analysis dataset contains 8,972 unique canonical polymer SMILES after merging train.csv with labeled supplements dataset1, dataset3 and dataset4. Counts exclude SMILES augmentation, test data, model weights and descriptor/install files.
2. Supplemental duplicate structures are averaged within each property. Existing primary labels are retained; supplements fill missing primary labels or add new structures. No pseudo-label source is included.
3. Labeled polymers are counted separately for each property. Missing percentages use the same merged dataset size as denominator. Mean and sample SD (ddof = 1) use only observed labels in original units, before target scaling.
4. Tg is reported in °C, following the competition documentation. Canonical SMILES preserve specified stereochemistry; this identity rule does not establish equivalence under all polymer repeat-unit shifts or periodic representations.
5. This table is recomputed from downloaded data using the author training notebook v1 and the supplied original inference canonicalizer. It is not independent proof of the exact membership of the historical 25 checkpoints; their output contains weights without row/fold manifests.
