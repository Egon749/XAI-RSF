# 执行入口与交付索引

本轮只读政策审计；不调用任何模型fit。优先阅读 [研究判断](RESEARCH_DECISION_ZH.md)、[标签审计](POLICY_LABEL_AUDIT_ZH.md) 和 [错误诊断](POLICY_ERROR_ANALYSIS_ZH.md)。所有公开数据均为规则元数据或聚合结果。

实际私有目录：`/home/oliver/MainDiskData/st_policy_alignment_v1_20261028T110421Z`。环境：`/home/oliver/.venvs/xai-baselines-20261027/bin/python`。不需安装训练依赖；使用现有numpy/pandas/pyarrow/scikit-learn/pytest，文本提取用pdftotext及本轮隔离解包的antiword。

复算已有冻结数据及已审查证据：

```bash
python experiments/st_policy_alignment_v1/run_diagnostics.py PRIVATE_RUN
python experiments/st_policy_alignment_v1/feature_availability.py PRIVATE_RUN
python experiments/st_policy_alignment_v1/finalize_evidence.py PRIVATE_RUN
POLICY_RUN=PRIVATE_RUN OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q experiments/st_policy_alignment_v1/tests
python experiments/st_policy_alignment_v1/verify_policy_outputs.py PRIVATE_RUN
```

`prepare.py`仅用于新的唯一私有目录，先创建input/audit/private/sources/results；原文件只读复制。`collect_sources.py`只对空sources目录执行，原件使用独占创建；antiword路径通过ANTIWORD_BIN及ANTIWORDHOME配置。`collect_event_announcements.py`读取已有官方stock lookup，不猜orgId，保存有界查询和失败。私有手工核对表 `event_manual_review_PRIVATE.json` 是本次审查输入，须随来源及哈希保留，不能用自动关键词替代；Git不含公司ID和公告摘录。再次抓取需新目录，不覆盖证据。

`origin_policy_context.parquet`是B表；`event_rule_evidence.parquet`为逐事件摘要，`event_rule_evidence_long.parquet`为一事件多条依据的长表。三个模型逐记录Top名单、事件提前量、公告原件/摘录和全部个体冲突候选均留私有目录。公共CSV不能用于反推公司模型参数。

`st_rule_versions.csv`的effective_to是事后版本结束元数据，未决明确标记；as-of解析不以未来废止信息截断历史知识。VERIFIED_TEXT_BOUNDARY只指已核查边界，整体适用性仍受例外及缺失事实约束。
