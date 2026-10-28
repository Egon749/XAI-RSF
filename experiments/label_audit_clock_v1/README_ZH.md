# 标签裁决与审计时钟核查 v1

先读[推进决定](NEXT_STEP_DECISION_ZH.md)，再读[标签裁决](LABEL_ADJUDICATION_ZH.md)、[映射核查](REASON_MAPPING_AUDIT_ZH.md)、[信息时间线](INTERNAL_CONTROL_TIMELINE_ZH.md)、[模型捕获对应](MODEL_CAPTURE_AND_INFORMATION_ZH.md)及[来源可行性](AUDIT_INPUT_FEASIBILITY_ZH.md)。执行边界在[协议](PROTOCOL_ZH.md)，实际测试在[验证记录](VALIDATION_ZH.md)。

私有运行目录：`/home/oliver/MainDiskData/label_audit_clock_v1_20261028T133649Z/`。所有逐公司证据与Parquet留该目录，Git只有匿名汇总。

重放现有证据（不会下载或拟合）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/oliver/.venvs/xai-baselines-20261027/bin/python \
experiments/label_audit_clock_v1/run_review.py \
--run /home/oliver/MainDiskData/label_audit_clock_v1_20261028T133649Z
```

运行依赖受保护的既有原始快照和私有 `audit_review_annotations.json`、`label_review_decisions.json` 等。annotation是原文核查结果，候选片段提取程序不会自动授予VERIFIED状态。不携带这些私有证据时，仅Git代码不足以重建逐公司结论。

```bash
LABEL_AUDIT_RUN=/home/oliver/MainDiskData/label_audit_clock_v1_20261028T133649Z \
/home/oliver/.venvs/xai-baselines-20261027/bin/python -m pytest -q \
experiments/label_audit_clock_v1/tests
```

只校验补丁：`validate_label_patch.py <私有运行目录>`。此工具不写修改后的标签。不得把补丁放进旧数据目录，也不得顺手启动拟合。

results目录提供冻结捕获、提前信息、配对捕获差、标签影响、映射范围及未决统计。未决统计的单位包括事件、文档、事件—财年—类型位置和证据行，不可相加。`execution_receipt.json`记录实跑结果及哈希；所在Git提交标识本版代码，服务器另存最终推送回执。
