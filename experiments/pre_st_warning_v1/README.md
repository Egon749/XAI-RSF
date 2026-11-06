# 正常企业首次财务困境预警迁移

本轮从本地 `pylib/csmar` 重建正常非金融A股公司的年度风险集。在2023、2024、2025年1月1日预测首次财务／审计类ST或*ST，12个月为主，24个月为次；其他首次ST及首次ST前直接退市为竞争事件。未知原因或未知公告时间按未解决状态删失，不设为正常。

固定迁移现有XAI-RSF结构：q≤2、深度4、共享分位切点、均衡25变量、目标事件公司支持保护2、直接父节点AJ收缩10、17/503/22三个种子各32棵。未来ST公告中的信息不能用于预警，因此重建75项事前输入（48财务、24市场、3质量），不直接复制ST后74变量矩阵。没有调参、筛变量、选种子或校准。保留原参考模型。

原始日行情实际从2015年开始，故可核验的年度入组从2016年开始。各截止日前的历史年度记录用于训练，按公司聚类抽样，随访在24个日历月或历史截止日前截断。同一公司可以作为早期训练记录与后期评价对象，但训练记录绝不包含预测截止日及以后的结局。

`frozen_protocol.json` 是拟合前协议；`feature_schema.json` 固定字段顺序。`run_location.json` 指向新建的私有共享运行目录。公司级数据、原始字段、切点、模型与预测只留私有目录。本轮使用当前导出的财务版本，已知更正按披露日限制；没有宣称历史原始版本全部通过PIT认证。2025年24个月性能保持NA。

复现入口：

```bash
python -m experiments.pre_st_warning_v1.finance --root /home/oliver/pylib/csmar --out NEW_RUN/prepared
hetero-run NEW_RUN/raw_tasks.jsonl
python -m experiments.pre_st_warning_v1.prepare --run NEW_RUN --root /home/oliver/pylib/csmar
python -m experiments.pre_st_warning_v1.stage --run NEW_RUN --stage pilot
hetero-run NEW_RUN/pilot_tasks.jsonl
python -m experiments.pre_st_warning_v1.stage --run NEW_RUN --stage formal
hetero-run NEW_RUN/formal_tasks.jsonl
python -m experiments.pre_st_warning_v1.stage --run NEW_RUN --stage ensemble
hetero-run NEW_RUN/ensemble_tasks.jsonl
python -m experiments.pre_st_warning_v1.analyze --run NEW_RUN
```

首次原始扫描清单由 `stage --stage raw` 创建。各次运行必须使用新的私有目录，数据与输出分开；不要重交已拟合成功的任务。若仅发布失败，使用原任务发布回执恢复，不重新训练。完整测试须配置仓库已有的 `CRFOREST_RSCRIPT` 与 `EW_RLIB`，同时限制BLAS为单线程。
