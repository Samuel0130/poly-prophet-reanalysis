# 外部基线工作区

`baseline_registry.md` 和 `baseline_registry.json` 是逐篇审计与纳入规则。

`results/Table3_fair_baselines.md` 是当前数据上可公平比较的第一批结果：当前 Poly-Prophet、均值预测、Morgan-Ridge、Morgan-RF。所有模型共享 `training/artifacts/outer_folds.csv`、目标特异标签和内部训练折选择。

`chen_rnn_tg/` 是 Chen et al. 2021 的公开仓库快照（commit `6a4f61554d8a9ce42dd99468dace793f5e02afca`）。其方法来自 PolyInfo 和原论文的随机划分，不能直接使用论文报告值与当前 OOF 结果比较。

`bioplastic_design/` 是 Kuenneth et al. 2022 的公开仓库快照（commit `069cc60701ae3ffec487d9bc1f7b24d44a95a905`）。README 明确说明预测器没有包含，因此只能作为文献背景。

`reported_literature_values.csv` 只保存论文报告值和可比性状态，不作为同数据 Table 3 的统计输入。
