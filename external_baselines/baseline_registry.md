# 外部基线登记（Table 3 方案）

登记原则：只有在同一重建数据、同一 grouped five-fold 划分、同一标签过滤、同一 MAE/RMSE 定义下重新训练的模型，才放入主比较表。论文原文的报告值单列，不能直接与当前结果混合排序。

| ID | 文献 | DOI | 原任务/数据 | 当前数据可迁移性 | 代码/可复现性 | 当前处理 |
|---|---|---|---|---|---|---|
| B0 | Mean predictor | — | 当前 NeurIPS 数据 | 直接 | 本地实现 | 必做最低基线 |
| B1 | Morgan fingerprint + Ridge/Random Forest | — | 当前 NeurIPS 数据 | 直接 | 本地实现 | 必做传统基线 |
| B2 | 纯 SMILES CNN | 与原模型相同输入 | 当前 NeurIPS 数据 | 直接 | 从原代码抽取 | 必做序列基线 |
| B3 | 纯 SMILES BiLSTM | 与原模型相同输入 | 当前 NeurIPS 数据 | 直接 | 从原代码抽取 | 必做序列基线 |
| B4 | Kuenneth et al. 2022, Bioplastic multitask DNN | 10.1038/s43246-022-00319-2 | PoLyInfo；22,731 条 homo/copolymer；13 属性；RMSE/R² | 只能方法重实现；数据不同、标签不同 | 公开仓库只有候选筛选，预测器未公开 | 不放直接 MAE 主表；可作为文献背景 |
| B5 | Chen et al. 2021, chemical language model | 10.3390/polym13111898 | Tg 专用聚合物数据 | 需确认数据及 tokenizer 后才能迁移 | 论文可得，未确认可运行仓库 | 候选 Tg 外部基线，待方法复核 |
| B6 | Parvez & Mehedi 2025, Pareto-SMILES | 10.3390/polym17131801 | 多性质 SMILES 任务 | 需确认标签和代码；不可直接套用报告值 | 公开论文，未确认仓库 | 候选多性质基线，待方法复核 |
| B7 | Zhang et al. 2024, Polymer-Unit Graph | 10.1021/acs.jctc.3c01385 | 有机聚合物半导体；Rg/目标定义可能不同 | 结构输入和目标需核对，不能默认适配 | ACS 正文/补充材料受限 | 仅候选 Rg/图基线 |
| B8 | Han et al. 2024, Multimodal Transformer | 10.1021/acsami.4c01207 | 聚合物多模态属性任务 | 需原始模态和标签；当前只有 SMILES/五性质 | 期刊正文/补充材料受限 | 仅候选多模态基线 |
| B9 | Aleb & Abu-Thabit 2025, TransTg | 10.1007/s00521-024-10532-4 | Tg 专用 Transformer | 可用 SMILES 重实现，但需精确 tokenizer/数据协议 | Springer 正文可查，代码未确认 | 候选 Tg 基线 |

## 主表的最小公平集合

先做 B0–B3：它们不依赖外部数据集，能回答“当前融合模型相对于简单基线是否有增益”。然后仅加入能够获得明确实现细节的 B5/B6/B9。B4 不适合直接重训当前五性质任务，因为原数据为 PoLyInfo、属性集合和评价指标均不同；B7/B8 需要原论文特有输入，缺失输入时不能声称公平复现。

## 统一评估协议

- 使用 `table1_rebuilt/merged_labeled_polymers.csv`。
- 每个性质单独使用非缺失标签；共享 `training/artifacts/outer_folds.csv` 的五折划分。
- 以 canonical non-isomeric SMILES 作为保守分组标识，组不可跨 fold。
- 外层测试集只评分一次；模型选择使用外层训练部分的内部 20% 分组验证集。
- 报告 OOF MAE、RMSE、R²、五折 MAE 均值 ± SD 和组级 bootstrap 95% CI。
- 传统基线的预处理只在每个训练折拟合，不能使用外层测试统计量。
- 论文报告值另列 `Reported literature value`，同时注明数据集、单位、指标和验证协议。

## 不能做的比较

不能把原论文 Table 2 的 Tg 4.02、Tc 0.0034、FFV 0.0022、Density 0.0041、Rg 0.225 与当前 OOF 结果直接排名；它们没有经过同数据同折验证，且 Tg 4.02 与原论文 Table 1 的 53.39 已存在内部不一致。
