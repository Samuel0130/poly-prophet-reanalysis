# 修订核查说明

本包以本地可核验的 OOF 结果为准，修订 `main.tex`。原论文中的 4.02 K (Tg) 和 0.225 Å (Rg) 未出现在可追溯的五折 OOF 文件中，且其数据集、划分和缩放流程无法确定，因此从定量比较中删除，并在正文说明。

已完成：
- 数据集组成、标签数、来源重叠、重复结构和缺失情况说明；
- 按去立体化 canonical SMILES 分组的五折划分；增强只用于训练折，标准化和早停只使用训练数据；
- 主模型 OOF 的 MAE、RMSE、R2、fold mean ± SD 和 polymer-group bootstrap CI；
- 七个消融模型和 30 个校正后的配对比较；
- 同数据、同折、同标签过滤的 mean、Morgan-Ridge、Morgan-RF、Chen RNN transferred、PU-GN adapted 基线；
- MolFormer domain-mismatch 的谨慎表述和 descriptor 实现缺陷披露；
- 明确这是五个独立的 property-specific regressors，不是一个多任务输出层。

仍明确标注为限制：本地可核验归档没有 PolyNC/polyBERT 的同协议结果，也没有相似度聚类重训结果；正文没有把它们伪装成已完成实验或数字结论。

主要结果（fold mean ± SD）：Tg 53.44 ± 4.79 K；FFV 0.00507 ± 0.00031；Tc 0.03108 ± 0.00291 W/(m·K)；density 0.02835 ± 0.00376 g/cm³；Rg 1.402 ± 0.144 Å。

结果来源：`training/results/performance_summary.csv`、`training/results/Table4_ablation.csv`、`training/results/ablation_comparisons.csv`、`external_baselines/results/Table3_fair_baselines.csv`。
