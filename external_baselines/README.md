# 外部基线工作区

`baseline_registry.md` 和 `baseline_registry.json` 是逐篇审计与纳入规则。

`results/Table3_fair_baselines.md` 是当前数据上可公平比较的第一批结果：当前 Poly-Prophet、均值预测、Morgan-Ridge、Morgan-RF。所有模型共享 `training/artifacts/outer_folds.csv`、目标特异标签和内部训练折选择。

`chen_rnn_tg/` 是 Chen et al. 2021 的公开仓库快照（commit `6a4f61554d8a9ce42dd99468dace793f5e02afca`）。其方法来自 PolyInfo 和原论文的随机划分，不能直接使用论文报告值与当前 OOF 结果比较。

`bioplastic_design/` 是 Kuenneth et al. 2022 的公开仓库快照（commit `069cc60701ae3ffec487d9bc1f7b24d44a95a905`）。README 明确说明预测器没有包含，因此只能作为文献背景。

`reported_literature_values.csv` 只保存论文报告值和可比性状态，不作为同数据 Table 3 的统计输入。

## 外部论文代码的当前数据训练

- `run_chen.py` 直接抽取作者 `getRNNmodel` 函数，使用论文最优参数：120 字符、15 维嵌入、两层双向 LSTM（每方向 60）、TimeDistributed Dense(30)、flatten、标量输出。当前 Keras 3 使用 torch 后端；词表和目标标准化只在训练部分拟合。Tg 是原任务，其他四个性质明确标为方法迁移。
- `vendor/purs/` 为 Polymer-Unit Graph 作者团队的公开代码快照。`vendor/provenance.json` 记录仓库与 commit；`vendor/torchgraphs/` 保存其旧依赖及许可。
- `prepare_pugraph.py` 使用作者 PURS 结构分解和 MACCS167 节点、8 维边特征，逐分子处理，避免结构单元索引由测试集决定。两个旧 pandas 链式写入用 `.loc` 修复；sum scatter 使用 PyTorch 原生 `index_add`。补丁在 `graph_compat.py`，下载代码保持原样。
- `run_pugraph.py` 直接调用作者 `SolubilityGN`，保留配置中的 3 层、hidden32、sum、dropout、Adam .001、L1 .0001。PURS 失败样本用完整重复单元作为单节点（MACCS167，无边），保留全部测试行。这是**适配版探索性基线**，不是原论文方法的忠实完整复现。
- `results/pugraph_conversion.csv` 记录 8972 个结构中 2236 个分解失败。失败比例随目标变化，Tc/Density/Rg 超过一半；这限制了 PU-GN 比较的科学解读。

运行（已有 `training/requirements.txt` 环境）：

```bash
uv pip install --python .venv/bin/python -r external_baselines/requirements-external.txt
.venv/bin/python external_baselines/run_chen.py
.venv/bin/python external_baselines/prepare_pugraph.py
.venv/bin/python external_baselines/run_pugraph.py
.venv/bin/python external_baselines/make_report.py
.venv/bin/python external_baselines/compare_papers.py
```

五折结果完成后才进入汇总表。`compare_papers.py` 核对共享 OOF 行、折和标签，生成 Word、Excel、逐样本配对误差和组 bootstrap CI。CI 条件于固定 OOF 模型，不覆盖重训随机性，未进行多重检验校正。

其他已核查论文：Kuenneth 仓库不包含预测器；TransTg 和 Han 暂未找到匹配的公开代码；Parvez 为分类任务且没有可核验代码链接。它们没有被虚构成回归实验结果。
