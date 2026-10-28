# 冻结协议与复现

授权范围为全部非金融行业、首次因财务／审计原因被ST或*ST之后的恢复／退市预测，2023与2024入组分别测试。“首次”是已有历史来源中的审计接受首次，不是完整上市生命周期已证明首次。先前已有非目标ST的企业不当作新的首次目标ST加入。

起点为首次目标ST公告日。终点1为先于完整恢复的最终终止上市公告X（全原因）；终点2为先于最终终止上市的完整恢复公告A。B/D/C/S/T都是中间状态，T不等于X；先恢复后退市者属于首次恢复。预测的是这两个首次转归的CIF，不是“最终必然退市概率”或撤销*ST但仍为ST的概率。两个CIF之和不必为1，余量是时限内未发生首次转归的概率。

每家公司仅一个起点，主时限24日历月、次时限12日历月。固定随访来源边界2026-09-03；训练信息边界严格早于2023-01-01，因此近期训练病例可被右删失。禁止用训练公司2023年后的结局补齐训练。2024不滚动吸收2023测试结局。特征仅取入组公告日期之前已经记录可得的内容；无日内时间的同日披露一律排除。

## 有限模型比较

三个配置、三个固定种子42/101/202，共9个正式拟合。所有模型为32棵树、最大深度6、75个入模字段；共同公司bootstrap。训练309家公司中的原始观察事件为退市14、恢复184。因样本从数万季度记录变为数百家公司，在拟合前冻结叶最小记录数10、公司组数5（原100/20），两种Python森林同时调整；不是看测试表现后的调参。

| 配置 | 冻结设置 | 比较边界 |
| --- | --- | --- |
| XAI-RSF q2 | 路径最多2变量；每树25/75变量子空间；共享分位数切点；最多5候选阈值；每叶总事件至少3 | 完整二阶CIF函数；未启用全森林格数上限或信息惩罚 |
| 同后端无限阶 | 不限路径变量；75变量可用；局部随机切点；其余共同预算 | 与q2同时差在阶数、覆盖组织和切点，不能单独估计q的因果贡献 |
| 标准RSF | randomForestSRC 3.9.0；logrankCR，两原因权重(1,1)；nodesize10，nodedepth6，mtry75，nsplit5 | 与Python后端不逐位等价；原生分裂和节点约束不同；缺失用仅训练列中位数补齐 |

另有训练Aalen–Johansen无协变量基线。Python采用原生缺失路由；R补齐不新增缺失指示维度。标准RSF是新任务的标准实现参考，并非经过充分调优的性能上界。旧M2预测首次ST，不能用于本次ST后转归，未重训或覆盖旧M2。

## 评价

分2023、2024与合并期重算AUC、AP、IPCW Brier；合并AP不平均两个年份。主要比较三种子指标均值及范围，范围不是置信区间；另列预先等权平均三种子预测形成的集成指标，二者不可混用。当前测试81家公司在12/24月均有完整终点判定，IPCW权重均为1；训练表观指标仍正确处理删失，且不是泛化性能。

删失时钟用精确日历月，事件时取G左极限，已知竞争事件作为该原因的非目标结局，未知结局不给端点评价权重。未做权重下限裁剪。记录等权与公司等权完全相同：每公司仅1条。Top20/25按每个入组年风险从高到低、向上取整；并列以固定sample_id顺序打破。此为全年回顾性核查容量比较，不是当时可部署的全年在线排名。无重复预警；提前3/6个月用公告日至终止公告日的日历跨度。

## 执行入口

拟合不可变提交：`db8e154d217d12faeda03d9d9f83e0e1195ab460`。代码归档、输入哈希和模型参数保留于 `/home/oliver/MainDiskData/xai_rsf_post_st_prognosis_v1_20261027_u6l00ebk`。`frozen_protocol.json` 在试运行前冻结，`results/pre_fit_protocol_freeze.json` 给出其哈希。新复现必须创建唯一私有目录，禁止覆盖本次运行。

```bash
# 在独立已提交的checkout与已存在隔离环境执行；RUN为新的私有目录
python -m experiments.post_st_prognosis_v1.prepare --run "$RUN"
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --repo "$REPO" --stage freeze
/home/oliver/.local/bin/hetero-submit "$RUN/pilot.jsonl" physical both
# 等待完成并使用真实作业号；先检查试运行再提交正式任务
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --stage pilot --job "$PILOT_JOB"
/home/oliver/.local/bin/hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_prognosis_v1.stages --run "$RUN" --stage formal --job "$FORMAL_JOB"
python -m experiments.post_st_prognosis_v1.evaluate --run "$RUN" --output "$RUN/reports_calendar_months"
python -m experiments.post_st_prognosis_v1.validate --run "$RUN" --reports "$RUN/reports_calendar_months"
python -m experiments.post_st_prognosis_v1.plot --reports "$RUN/reports_calendar_months"
```

`prepare` 要求调用前创建空的 `input/ audit/ private/ reports/ results/`；输入源路径和解释器为本服务器配置，不能假设在其他机器存在。`stages freeze` 要求干净Git工作区。若另用hetero-run sidecar，先核对物理核布局；本轮明确使用physical/both提交。只有计算完成、结果发布及哈希验证全通过才记成功。`validate` 有本轮样本及156条完整端点检查，换队列时需显式重新审计这些断言，不能删除失败断言强行通过。
