# 核查容量和校准

Top20%按2025年度53家公司向上取整，核查11家；Top25%核查14家。按入组时固定12个月CIF排序，并列按固定sample_id排序。以下为固定三种子集成，不是挑选最佳种子。

| model | top_fraction | checked_companies | captured_events | all_target_events | event_coverage | known_non_target_checked | unknown_checked | identified_at_least_3_calendar_months_before_termination | identified_at_least_6_calendar_months_before_termination |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XAI-RSF q2 | 0.2000 | 11 | 1 | 5 | 0.2000 | 10 | 0 | 1 | 1 |
| XAI-RSF q2 | 0.2500 | 14 | 1 | 5 | 0.2000 | 13 | 0 | 1 | 1 |
| 同后端无限阶森林 | 0.2000 | 11 | 1 | 5 | 0.2000 | 10 | 0 | 1 | 1 |
| 同后端无限阶森林 | 0.2500 | 14 | 1 | 5 | 0.2000 | 13 | 0 | 1 | 1 |
| 标准RSF（randomForestSRC） | 0.2000 | 11 | 1 | 5 | 0.2000 | 10 | 0 | 1 | 1 |
| 标准RSF（randomForestSRC） | 0.2500 | 14 | 1 | 5 | 0.2000 | 13 | 0 | 1 | 1 |

三个模型在两个核查容量下都只捕获1/5个退市事件（20%覆盖）；捕获事件满足提前3个月及6个月条件。Top25增加核查公司，没有增加捕获数量。相同捕获数不表示三个模型选中了完全相同的公司。已知非目标包含恢复和时限内无首次转归，不能都称为正常；本队列未知0、重复预警0。

这是全年回顾性容量比较，不是假定年初可以预知全年入组公司池的在线预警政策。全部种子核查结果保存在CSV，未只留集成或成功种子。概率校准五等分表见 `results/calibration.csv`；只有5/3个事件，分箱发生率波动大，不实施基于本测试集的再校准。
