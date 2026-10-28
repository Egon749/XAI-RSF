# 原始年报确认口径：C组有限重训

研究者先确认现有年报数值视为原始版本，再明确要求重新训练。本轮只重训已准备的83列C组，q2和标准RSF、2023和2024各三个固定种子42/101/202，共12个正式模型，加4个四树数值试运行。原因权重、树数/深度、叶节点限制、共享切点、变量覆盖、bootstrap、叶内AJ和平均方式均不改变。

同来源年报在训练与验证中按已有首次公告日对齐，更正日期保留作审计信息；季度报告不变。原始数值没有编辑；财务报告选择或同比可用性变化影响训练38/41家公司、测试3/1家公司。此前25份测试年报解除版本限制，但21家仍采用较新第一季报。来源状态为研究者确认接受，不等同独立逐值PDF核验。

两折训练309/345家，14/26个24个月先退市事件；测试36/45家，12/20个先退市事件。与上一版C组的公司、时间、标签逐数组一致，原模型/预测只读复用。主结果先平均三种子预测再评价；两年及合并AP分别重算，Top20为主、Top25补充，12个月为补充。全部种子、不利结果和失败均保留。年度完整队列排序不是在线预警策略。

完整解释保持CIF尺度上的主效应+二元交互；检查原生时间概率约束、重载、全部节点统计及96树完整函数。没有校准、叶收缩、额外权重/阶数/超参数搜索或2025入组评价。测试结局此前已接触，属于探索性比较；本轮检验来源口径的影响，不是独立算法创新。

执行顺序：唯一私有目录建audit/input/evaluation_input/private/reports/results后：

```bash
python -m experiments.post_st_annual_accepted_refit_v1.prepare --run "$RUN"
# 完整测试包含独立R通过并提交代码后，复用已验证的冻结和任务入口：
python -m experiments.post_st_next_open_info_v1.stages --stage freeze --run "$RUN" --repo "$REPO"
hetero-submit "$RUN/pilot.jsonl" p both
python -m experiments.post_st_next_open_info_v1.stages --stage pilot --run "$RUN" --job "$PILOT_JOB"
hetero-submit "$RUN/formal.jsonl" p both
python -m experiments.post_st_next_open_info_v1.stages --stage formal --run "$RUN" --job "$FORMAL_JOB"
python -m experiments.post_st_annual_accepted_refit_v1.analyze --run "$RUN"
python -m experiments.post_st_next_open_info_v1.audit --run "$RUN"
python -m experiments.post_st_next_open_info_v1.field_audit --run "$RUN"
```

现有Slurm的p布局由其拓扑逻辑选择oliver P核及lemon物理核，不硬编码CPU；任务单线程、4GiB地址空间限制、NFS及校验回传。12个独立任务不足以填满32个物理工作核，不为提高利用率重复模型。本布局不称为实测最快。
