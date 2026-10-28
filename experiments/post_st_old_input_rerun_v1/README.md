# 旧C输入复跑

按用户要求恢复年报确认前的旧C输入，同时用于训练和测试。使用原83列、同一公司与标签、同一下一交易日开盘前起点及24个月主期限，q2和标准RSF、2023/2024、种子42/101/202，共12个新拟合。原模型仅作为复现对照，不代替此次训练。不新增试运行：相同实现、旧输入和预算已有成功的小试与正式执行记录。

本轮明确是看到新旧结果后由用户选择的旧输入复现，不是新的盲测或额外独立验证。此前研究者接受年报为原始版本的声明、接受口径下的完整结果和下降记录全部保留；此次仅恢复指定输入快照。不得由本轮推断旧输入在新样本上必然更好。

```bash
python -m experiments.post_st_old_input_rerun_v1.prepare --stage prepare --run "$RUN"
# 完整测试及独立R检查通过、提交代码后：
python -m experiments.post_st_old_input_rerun_v1.prepare --stage freeze --run "$RUN" --repo "$REPO"
hetero-submit "$RUN/formal.jsonl" p both
python -m experiments.post_st_next_open_info_v1.stages --stage formal --run "$RUN" --job "$JOB"
python -m experiments.post_st_old_input_rerun_v1.analyze --run "$RUN"
python -m experiments.post_st_next_open_info_v1.audit --run "$RUN"
python -m experiments.post_st_next_open_info_v1.field_audit --run "$RUN"
```

冻结输入与结果分目录，通过NFS及校验回传。两机现有p布局选择oliver P核及lemon物理核，每个任务单线程4GiB地址空间限制。没有硬编码CPU，也没有为使用率添加重复种子。模型、企业预测、实际阈值和完整函数参数保留服务器，仅发布汇总、代码和哈希。
