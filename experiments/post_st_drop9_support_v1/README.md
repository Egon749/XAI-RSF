# 74项输入＋退市支持保护：固定匹配实验

本实验补齐83/74输入 × 支持保护关/开中的74＋开单元，不更换当前研究参考模型。固定三个年度、三个种子；正式9次森林拟合，另1次4树数值试运行。raw森林各自派生直接父节点AJ10收缩模型。公司与模型工件只存于 `run_location.json` 指向的唯一私有目录。

执行入口（仓库根目录，现有Python研究环境）：

```bash
python -m experiments.post_st_drop9_support_v1.run prepare
# 完整pytest包括R检查通过后，提交协议与代码
python -m experiments.post_st_drop9_support_v1.run freeze
python -m experiments.post_st_drop9_support_v1.run pilot
python -m experiments.post_st_drop9_support_v1.run formal
python -m experiments.post_st_drop9_support_v1.run freeze_analysis
python -m experiments.post_st_drop9_support_v1.run ensemble
```

准备及发布入口采用一次性目录；复现需指定新的唯一运行目录，不能在原结果目录重跑。训练和函数导出任务由两节点Slurm池执行，每项单线程，4GiB地址空间限制。已保存的83原模型、74原模型和83支持模型复用，不重复拟合。

`frozen_protocol.json` 是本轮冻结边界；`PARAMETER_DIFFERENCES.csv` 核对新74模型相对于原74模型只有 `split_min_target_groups: 0→2` 变化。支持条件是子节点中退市公司数为0或至少2，不保证每个叶子有退市。

统一评价与报告：

```bash
python -m experiments.post_st_drop9_support_v1.analyze metrics
python -m experiments.post_st_drop9_support_v1.analyze functions
python -m experiments.post_st_drop9_support_v1.report
```

阅读顺序：`RESULTS_ZH.md` → `TERMINATION_COMPARISON.csv` → `METRICS_ENSEMBLE.csv`（两原因、raw/收缩、年内与pooled）→ `PAIRED_BOOTSTRAP_95.csv` → `NUMERICAL_COMPLEXITY.csv`。`METRICS_ALL.csv` 保留全部单种子。`MODEL_HASHES.csv` 提供模型、预测和函数文件哈希；公司明细与阈值不发布。

复用的历史74 raw合并预测参与统一评价；其三种子raw函数工件均复用，但没有为其额外编译96树raw函数表。主分析的全部四臂收缩96树函数表均有完整计数与验证。三种子稳定性只反映算法随机性。
