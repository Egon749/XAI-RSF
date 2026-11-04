# 近期 ESWA 方法在 ST 后退市排序中的对比

[中文结果与年度表](RESULTS_ZH.md) · [论文表述](METHOD_DECISION_ZH.md) · [原文与算法映射](PAPER_METHOD_MAPPING.md) · [冻结协议](FROZEN_PROTOCOL.json) · [六幅图](figures/) · [回执](execution_receipt.json)

基准为 969bfdf97bdbc6468235376f18deb744c602dcc6。主模型沿用已保存的 74 项、C039、96 树，另报告同配置的 42/101/202 组合。没有重新训练森林，也没有变更当前模型指针。第一份协议在 500930d 提交。收到 Chen 全文、尚未拟合任何真实分类器时，修订了原论文的 IPSO、全群重置、F1 归一与 GA 约束并再次冻结在 9ff50a0。

## 复核材料

- **来源与时间对齐**：PAPER_METHOD_MAPPING、ORIGINAL_VS_ADAPTED_METHODS、PROBOOST_IMPLEMENTATION_AUDIT、STACKING_IMPLEMENTATION_AUDIT、DATA_ALIGNMENT_AUDIT、CENSORING_AND_LABEL_AUDIT、MATURITY。原文 PDF 和逐公司来源不纳入仓库。
- **训练与实现**：SEARCH_BUDGET 按真实拟合计；TRAINING_LOG 每行记录一次实际操作（包括不拟合模型的搜索结束/回退记录），不可直接把行数当拟合数。HYPERPARAMETERS 记录最终超参数；MODEL_PARAMETERS 记录部署及外层元验证折方案；完整 F1 权重、私有折索引和模型仅保留在服务器。SMOTE_AUDIT 的合成行次数不是独立公司数。SELECTED_FEATURES 逐项列出年度最终/元验证折方案，SELECTION_STABILITY 提供 Jaccard。RUNTIME_AND_COMPLEXITY 按原模型或新任务分别计时。FIT_ACCOUNTING 另计被替代任务。
- **比较与验证**：METRICS_BY_YEAR、POOLED_METRICS、TOP20_CAPTURE、PAIRED_COMPARISONS、SCORE_DISTRIBUTIONS、MISSINGNESS_SENSITIVITY、TRAIN_G_RANKING_AUDIT、INTERPRETABILITY_COMPARISON、PROBOOST_PATHS、REFERENCE_VALIDATION、NUMERICAL_CHECKS、OOF_CHECKS、PUBLICATION_CHECKS。跨年合并 AP、AUC 从全体得分重算。2025H24 的真实评分只存私有目录，公开分布不含性能。区间是固定预测下的条件区间。SUPERSEDED_JOBS 专列初次回退错误及保留成本。

## 运行

在原服务器部署环境中，在新的唯一私有共享目录设置 ESWA_RUN。使用现有研究 Python、R 和两个计算节点可读的训练依赖。任务级启动脚本依据主机选解释器；两节点均需核验 numpy/scipy、scikit-learn 与共享依赖可加载。集群实际放置以回执为准。ESWA_RUN/deps 内使用 xgboost-cpu 3.4.1、lightgbm 4.7.0、hyperopt 0.3.0，和回执列出的依赖。不要改写既有私有目录；不需要提交训练数据或模型。

```bash
m=experiments.post_st_eswa_baselines_v1
export PYTHONPATH="$ESWA_RUN/deps:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m "$m.stage"
python -m pytest -q
python -m "$m.schedule"
python -m "$m.validation"
python -m "$m.evaluate"
python -m "$m.audit"
python -m "$m.weight_audit"
python -m "$m.delivery_stats"
python -m "$m.figures"
python -m "$m.report"
```

schedule 使用已提交的不可变代码快照，按年份、时限、算法和训练权重口径分为 24 个独立作业；正式分类器只读取训练快照与无标签评价 X、对齐键。评估另从既有研究读取冻结结局。实际第一批代码为 ada7910；发现 GA 支持不足时的等权回退缺陷后，14db16a 只重算 2023/H12 Stacking 的两种口径（schedule --repair），没有改动协议。直接用最终代码从零复跑只需首批 24 个作业。已有私有输出可只执行验证和报告命令。统计训练负担时，区分正式采用的 24 个部署预测器、内部拟合，以及被纠正的 2 个旧部署预测器。
