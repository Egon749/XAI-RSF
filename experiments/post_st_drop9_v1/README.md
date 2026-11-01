# 固定删除九项疑似冗余变量的匹配实验

已完成3/3次新森林拟合。24个月AUC 0.770→0.694，AP 0.782032→0.693633，Brier 0.339557→0.343057，Top20捕获8/20→7/20；12个月排序指标有所改善。保留原模型，不追加筛选。详见[中文结果](RESULTS_ZH.md)、[数据核验](DATA_ALIGNMENT_AUDIT_ZH.md)、[训练诊断](TRAINING_DIAGNOSTICS_ZH.md)、[论文段落](METHODS_RESULTS_ZH.md)及[执行回执](execution_receipt.json)。

相对保留的83变量参考模型，只将固定九项字段删除，其他训练配置保持不变。使用同一2024冻结训练集（2012—2023原点、信息严格早于2024-01-01），新增三个种子各32树，原始AJ森林保存后派生直接父节点AJ强度10版本。

删除名单固定于上一轮截至2024年初的历史诊断，见frozen_protocol.json及feature_map.csv。新模型74列，q2、深度4、每树均衡25变量、mtry75、共享5切点、叶设置10/5/3、原因权重(1,1)、时间权重1均不变。mtry的旧上界检查需允许75>74，节点仍使用原有min(mtry,可用变量数)；单元测试验证与实际有效上限一致，不改变旧配置行为。

维数变化会改变均衡覆盖周期4→3、具体变量组合及25/83→25/74的资格比例。公司bootstrap严格配对；不宣称每棵树候选变量相同，也不将结果全部归为删除“纯噪声”的因果收益。

仅评价2024，主指标为24个日历月内先最终退市：AUC、AP、IPCW Brier、年度Top20捕获（45家中核查9家）。三个种子逐公司先平均后评价，单种子与12个月/恢复次指标完整保留。当前模型、旧结果、输入数据均不覆盖，无追加调参或校准。

新运行入口（RUN为新建独立空私有目录）：

```bash
python -m experiments.post_st_drop9_v1.prepare --stage prepare --run RUN
# 完成全套测试并提交冻结代码后：
python -m experiments.post_st_drop9_v1.prepare --stage freeze --run RUN
/home/oliver/.local/bin/hetero-run RUN/formal.py
python -m experiments.post_st_ablation_v1.stages --stage formal --run RUN --job JOB --report REPORT
python -m experiments.post_st_drop9_v1.analyze --run RUN
python -m experiments.post_st_drop9_v1.deliver --run RUN
```

正式冻结运行提交为5558513eaa188b4bc128125e5776cb4d581c6a1d；交付脚本与汇总报告在后续结果提交中补充。运行目录中的immutable input/code快照不变，全部结果来源于该快照。运行入口以新空目录为目标，不可覆盖本次现存目录；重放评价可使用私有保存预测，不需要再次训练。

实验为已暴露2024队列上的探索性匹配比较，不能声称新的独立验证。财务原始版本/PIT限制继承。公司明细、预测、模型、真实切点和函数参数仅留服务器；仓库只发布代码、冻结配置、汇总诊断和哈希。
