# 旧C输入复跑

已完成12个新拟合，预测与原C组完全一致（最大差0）；120项完整测试通过。q2的2023/2024 AUC为0.736111/0.752000、AP为0.576727/0.758339；Top20合计12/32，Top25为15/32。完整解释继续精确，概率低估和交互稳定性限制不变。计算结论EXACT_REPRODUCTION，研究判断仍为INCONCLUSIVE。

[主结果](RESULTS_ZH.md)、[双结局与标准RSF比较](PREDICTION_COMPARISON_ZH.md)、[数据对齐](DATA_ALIGNMENT_AUDIT_ZH.md)、[完整解释](INTERPRETABILITY_COMPARISON_ZH.md)、[执行与复现核验](TRAINING_DIAGNOSTICS_ZH.md)、[研究判断](METHOD_INNOVATION_DECISION_ZH.md)、[全部指标](results/metrics.csv)、[回执](execution_receipt.json)。先前[接受年报口径的结果](../post_st_annual_accepted_refit_v1/README.md)保留。

按用户要求恢复年报确认前的旧C输入，同时用于训练和测试。使用原83列、同一公司与标签、同一下一交易日开盘前起点及24个月主期限，q2和标准RSF、2023/2024、种子42/101/202，共12个新拟合。原模型仅作为复现对照，不代替此次训练。不新增试运行：相同实现、旧输入和预算已有成功的小试与正式执行记录。

本轮明确是看到新旧结果后由用户选择的旧输入复现，不是新的盲测或额外独立验证。此前研究者接受年报为原始版本的声明、接受口径下的完整结果和下降记录全部保留；此次仅恢复指定输入快照。不得由本轮推断旧输入在新样本上必然更好。

```bash
python -m experiments.post_st_old_input_rerun_v1.prepare --stage prepare --run "$RUN"
# 完整测试及独立R检查通过、提交代码后：
python -m experiments.post_st_old_input_rerun_v1.prepare --stage freeze --run "$RUN" --repo "$REPO"
hetero-submit "$RUN/formal.jsonl" p both
python -m experiments.post_st_next_open_info_v1.stages --stage formal --run "$RUN" --job "$JOB"
python -m experiments.post_st_old_input_rerun_v1.analyze --run "$RUN"
python -m experiments.post_st_old_input_rerun_v1.reproduce --run "$RUN"
python -m experiments.post_st_next_open_info_v1.audit --run "$RUN"
python -m experiments.post_st_next_open_info_v1.field_audit --run "$RUN"
# RDS比较入口compare_r_objects.R读取由reference_models与formal清单建立的私有路径表。
# 再次完整测试通过后：
python -m experiments.post_st_old_input_rerun_v1.deliver --run "$RUN" --dest "$REPO/experiments/post_st_old_input_rerun_v1"
```

冻结输入与结果分目录，通过NFS及校验回传。两机现有p布局选择oliver P核及lemon物理核，每个任务单线程4GiB地址空间限制。没有硬编码CPU，也没有为使用率添加重复种子。模型、企业预测、实际阈值和完整函数参数保留服务器，仅发布汇总、代码和哈希。
