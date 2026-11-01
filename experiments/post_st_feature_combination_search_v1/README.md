# 历史信息约束下的特征组合搜索

参考提交722089d。保留83列q2/depth4/shared/balanced25/parentAJ10模型，扩展训练前特征集合搜索。冻结协议在`frozen_protocol.json`；历史候选信息日期在`HISTORICAL_CANDIDATE_AVAILABILITY.csv`。

完整枚举83→历史74/75之间的冗余字段组合，再进行两轮多起点的单变量增删、审计整块及相关组代表替换。74/75和83在每轮都保留为起点；中间步骤不要求预测改善，最终候选仍须通过原保护条件。仅小子空间完整枚举，不宣称全83变量的全局最优。2023完整74名单不得作为来自当时诊断的先验。

各候选以三个种子各32树真实重训、raw直接父AJ收缩及完整函数核验；可验证的历史同配置工件按哈希复用。选择器只读历史前向验证。最终外层2023–2025均为已暴露探索性比较，2025的24个月无支持保持NA。最终固定集合50次公司重训验证的是条件模型/函数稳定性，没有重复整个扩大搜索。

使用独立新目录复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/oliver/.venvs/xai-baselines-20261027/bin/python \
  -m experiments.post_st_feature_combination_search_v1.run_pipeline \
  --run /home/oliver/MainDiskData/xai_rsf_combinations_REPRO_UNIQUE
```

训练入口自动生成不可变代码快照、任务清单和`.hetero.json`，使用两节点已有调度和共享输入。每项输出独立，哈希回传成功才计完成；失败时保留工件，不自动重训已计算模型。公司数据、模型参数、真实阈值、逐公司预测/解释均保留私有目录。
