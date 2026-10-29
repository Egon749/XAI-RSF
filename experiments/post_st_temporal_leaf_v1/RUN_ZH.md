# 复现与交付位置

仓库分支：`research/xai-rsf-post-st-prognosis-v1`。已完成运行：`/home/oliver/MainDiskData/xai_rsf_temporal_leaf_20261028__kvhg19v`。原结果：`/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj`。本轮拟合代码SHA：`affddbe861a182d601ab4ff4bb6211872f6f1508`；报告与只读聚合脚本可在后续交付提交中追加，不改变冻结训练快照。

完整基础森林配置见[2023](configs/outer_2023.json)、[2024](configs/outer_2024.json)；worker仅按冻结协议替换种子、小试树数，或在leaf20臂替换min_leaf。时间加权不更换这些配置。

执行顺序为`diagnose.py` → `check_step0.py` → `prepare.py --stage prepare` → 全量测试及冻结代码提交 → `prepare.py --stage freeze` → 小试 → 历史验证 → `stages.py --stage select` → 正式任务 → `evaluate.py` → `summarize.py` → `figures.py`/`report.py`。代码入口均在本目录，运行模块时工作目录为仓库根。

`prepare.freeze`生成独立任务JSONL及 `.py.hetero.json` sidecar；本次实际由 `/home/oliver/.local/bin/hetero-run RUN/pilot.py`、`RUN/validation.py`、`RUN/formal.py` 提交。任务明确从DATA_DIR读，只向OUTPUT_DIR写，由池管理计算与NFS发布，成功后再检查每个输出哈希。复现实验应创建新的唯一RUN，不能覆盖既有目录；模型已存在时只读复核，不为重复导出重新训练。

已完成的只读评价入口（输出目录必须新建；勿覆盖现有证据）：

```bash
python -m experiments.post_st_temporal_leaf_v1.evaluate --run RUN
python -m experiments.post_st_temporal_leaf_v1.summarize --run RUN
python -m experiments.post_st_temporal_leaf_v1.figures --run RUN --output FIGURES
python -m experiments.post_st_temporal_leaf_v1.report --run RUN --output PUBLIC_REPORT
```

`evaluate.py`先要求正式任务通过及历史选择冻结，再读取2023/2024标签；它不选择新参数。`summarize.py`增加按历史选择规则逐年组合的合并指标，合并AP重新计算。CSV含每个种子、三种子平均预测、全部候选和两原因/两时点。`report.py`仅显式复制reports/evaluation/summary的汇总表，不复制公司数据、函数参数或模型。

全量测试须设置 `CRFOREST_RSCRIPT` 与 `EW_RLIB`（路径见environment_versions.json），并限制OPENBLAS/OMP/MKL为1；最终139 passed、0 skipped。PNG/PDF为汇总研究图，已人工查看渲染；CSV可逐项复算。所有输入哈希、模型/预测/函数哈希及保存重载结果可核验，私有参数和企业明细需在服务器读取。
