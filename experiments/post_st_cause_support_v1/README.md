# 原因分裂与退市支持保护：有限匹配消融

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

所有企业记录、预测、函数值、真实切点、森林参数保留服务器私有目录。2023—2025已经暴露，本轮为探索性机制比较，不是新独立验证。

已完成：36次正式拟合、2次数值试运行、24个完整平均预测器编译验证。主结果见RESULTS_ZH.md；数值验证、raw次分析、两原因双时点、配对区间、复杂度、函数稳定性及全部模型哈希均保留。结论是继续保留当前参考模型。
