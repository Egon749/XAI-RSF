# 原因分裂与退市支持保护：有限匹配消融

**本文件夹是本轮结果和代码的统一入口。** 已完成四臂消融、2023/2024退市支持保护排序图及逐企业排序表，当前参考模型保留。本次整理没有新增森林训练，也没有混入另一个正在进行的仿射校准实验。

| 内容 | 入口 |
| --- | --- |
| 中文完整结果、四臂比较与研究判断 | [RESULTS_ZH.md](RESULTS_ZH.md)、[METHOD_INNOVATION_DECISION_ZH.md](METHOD_INNOVATION_DECISION_ZH.md) |
| 2023/2024全部企业风险排序图及数据 | [rankings/README_ZH.md](rankings/README_ZH.md) |
| 逐企业排序CSV（162条公司—时限记录） | [rankings/ranked_companies.csv](rankings/ranked_companies.csv) |
| 两年度12/24个月完整排序PDF | [rankings/support_risk_rankings_2023_2024.pdf](rankings/support_risk_rankings_2023_2024.pdf) |
| 年度、合并、两原因、两时限与全部种子结果 | [METRICS_ENSEMBLE.csv](METRICS_ENSEMBLE.csv)、[METRICS_ALL.csv](METRICS_ALL.csv) |
| 数值、函数复杂度、稳定性及事件支持 | [TRAINING_DIAGNOSTICS_ZH.md](TRAINING_DIAGNOSTICS_ZH.md)、[INTERPRETABILITY_COMPARISON_ZH.md](INTERPRETABILITY_COMPARISON_ZH.md) |
| 冻结协议与数据审计 | [frozen_protocol.json](frozen_protocol.json)、[DATA_ALIGNMENT_AUDIT_ZH.md](DATA_ALIGNMENT_AUDIT_ZH.md) |
| 拟合、预测、评价、完整解释核验代码 | 本目录的[run.py](run.py)、[worker.py](worker.py)、[analyze.py](analyze.py)、[verification.py](verification.py)、[ensemble.py](ensemble.py) |
| 森林核心实现、改动补丁及测试快照 | [implementation/README_ZH.md](implementation/README_ZH.md) |
| 仅从保存预测重画排序图 | [rankings/plot_saved_rankings.py](rankings/plot_saved_rankings.py) |
| 环境、模型哈希与执行回执 | [ENVIRONMENT.json](ENVIRONMENT.json)、[MODEL_HASHES.csv](MODEL_HASHES.csv)、[execution_receipt.json](execution_receipt.json) |
| 本次逐企业公开授权与文件哈希 | [rankings/publication_receipt.json](rankings/publication_receipt.json)、[BUNDLE_MANIFEST.json](BUNDLE_MANIFEST.json) |

结果要点：单独支持保护在2023年12/24月AUC/AP均提高；2024年12月AUC与Top20捕获提高但AP下降，24月略有损失。它应视为有条件收益的候选版本，尚未替换统一参考。2025年24月无足够随访支持，指标仍为NA。详见完整报告。

参考提交51e58f011dae19caba932659d98209a8658a0158。冻结前完整测试通过后提交代码与协议，随后运行数值试验和正式批次。当前研究参考不变。

四组为baseline、quadratic、support、quadratic_support；配置与逐参数差异见frozen_protocol.json和PARAMETER_DIFFERENCES.csv。正式36次raw拟合、36个父节点收缩派生版本；另2次4树数值试运行。基线9次是明确记录的重现，不称为历史工件复用。

新增分裂分数是原因标准化偏差的平方加权均值开方，消除跨原因有符号抵消，同时改变标准化方法；不是仅改变符号组合的单因素证明，不是卡方检验。退市保护要求各子节点独立退市公司数为0或至少2，禁止单家公司重复bootstrap制造支持。它不保证每叶有退市，不增加真实事件，也可能阻止有用的小群体划分。

执行入口（仓库环境；私有路径见run_location.json）：

```bash
python -m experiments.post_st_cause_support_v1.run prepare
# 完整pytest（包含独立R验证），提交冻结协议与代码
python -m experiments.post_st_cause_support_v1.run freeze
python -m experiments.post_st_cause_support_v1.run pilot
python -m experiments.post_st_cause_support_v1.run formal
# 提交只读评价/合并验证代码，冻结单独分析快照；不修改训练快照
python -m experiments.post_st_cause_support_v1.run freeze_analysis
python -m experiments.post_st_cause_support_v1.run ensemble
python -m experiments.post_st_cause_support_v1.analyze metrics
python -m experiments.post_st_cause_support_v1.analyze functions
python -m experiments.post_st_cause_support_v1.analyze mechanism
python -m experiments.post_st_cause_support_v1.deliver
python -m experiments.post_st_cause_support_v1.final_audit
```

仅在全新运行目录执行prepare/freeze；已执行目录不能覆盖。launch.py为节点解释器适配，hetero-run调度独立年/臂/种子。输入NFS只读，测试结局不送入拟合进程；结果由pool核验并发布。失败保留，不自动重训。后续analysis入口只读取保存模型和预测。

研究者随后明确授权发布本轮退市支持保护模型2023/2024的逐企业排序及预测图，授权范围和哈希见rankings/publication_receipt.json。原始财务输入、拟合参数、真实切点及授权范围之外的企业预测继续保留服务器私有目录。2023—2025已经暴露，本轮为探索性机制比较，不是新独立验证。

已完成：36次正式拟合、2次数值试运行、24个完整平均预测器编译验证。主结果见RESULTS_ZH.md；数值验证、raw次分析、两原因双时点、配对区间、复杂度、函数稳定性及全部模型哈希均保留。结论是继续保留当前参考模型。
