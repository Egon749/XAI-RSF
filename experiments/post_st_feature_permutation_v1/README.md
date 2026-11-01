# 历史模型的特征置换诊断

本目录与旧 `post_st_permutation_v1`（根节点标签置换）不同。本轮不重训、不重新校准、不删变量、不改变当前参考模型。

复用2019—2023各3个种子的旧C、83列、q2、深度4、共享分位数切点、均衡25变量资格、直接父节点AJ强度10的模型。2023模型用于对2023企业做历史样本外预测；2024参考模型原本已包含2023训练企业，但其在样本预测不用于筛选。

截止分别为2023-01-01和2024-01-01。仅读取相应冻结训练快照和历史滚动模型；不读取2023/2024最终测试标签。2023企业在后一截止前最长仅约8.7个月，不能冒充12/24个月完整验证。

协议、代码和输入哈希在置换前冻结。128次共同企业置换、1000次共同公司重抽样；以逐年三个种子平均预测为单位。主要指标为24个月先最终退市，恢复与12个月为次要保护性检查。相关组仅由对应训练X定义。所有原生二元项均检查，联合置换不等同于纯交互增量；去除原生二元项仅产生诊断分数。

复现入口（RUN必须是新建的空私有目录；不要复写已完成运行）：

```bash
python -m experiments.post_st_feature_permutation_v1.prepare --stage prepare --run RUN
# 核对数据审计，运行完整测试后提交冻结代码。
python -m experiments.post_st_feature_permutation_v1.prepare --stage freeze --run RUN
/home/oliver/.local/bin/hetero-run RUN/formal.py
# 核验调度器报告与全部发布SHA256；JOB和REPORT取实际返回值。
python -m experiments.post_st_ablation_v1.stages --stage formal --run RUN --job JOB --report REPORT
python -m experiments.post_st_feature_permutation_v1.analyze --run RUN
```

变量名单是探索性的保留/疑似冗余或噪声/证据不足建议，不是新的模型，也不能宣称筛选提高了AUC。原始财务版本PIT限制仍继承。完整结果与实际执行凭证在运行后交付。
