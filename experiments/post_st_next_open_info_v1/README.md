# 下一交易日开盘前：有限信息集实验

这是已接触2023／2024结局后的探索性实验。起点移动到入组公告日之后的下一个全市场交易日开盘前，同步重建风险集、两年随访和年度训练截止；保留原390家非金融企业的行业口径。不是将预测时刻虚构为已知的公告小时，也不是等待停牌公司复牌。

固定三组：`market_clock`保留旧48项财务值，更新24项市场变量和3项质量字段；`financial_refresh`再使用新起点前最新合格财务快照；`announcement_fields`再加公告警示类型、上一年度审计意见和持续经营非标标记，共83列。所有快照均要求公告与已记录的更正日期早于起点，增长／变化项检查同比组成数据日期。已知未来更正的版本不回填。**现有来源没有独立逐值核实的原始财务版本，本轮不是完整PIT修复，也不是“原始年报已恢复”的证明。** 新选择的最新报告可能是第一季报；年报缺口单独披露。

q2及标准randomForestSRC各用32树、深度6、同公司bootstrap，种子42／101／202；其他数值预算继承。q2每树25个候选输入、共享切点、路径至多二阶、叶内AJ与平均方式不改。75→83列改变变量覆盖比例，因此附加字段的效果包含搜索候选组成变化，不能当作某一字段的纯因果效应。审计意见哑变量有确定性冗余，不宣称统计可辨识性；缺失／冲突不编码成标准意见。

两年度×三信息集×两模型×三种子＝36个正式任务，另4个四树试运行。预定主结果为三种子预测平均后评价24个月首次最终退市X／完全恢复A，补充12个月。训练2023折含2012—2022入组、信息截止2023年初；2024折推进至2012—2023入组、信息截止2024年初。后续覆盖截至原证据2026-09-03。未知结局保持删失，先恢复后退市仍属于首次恢复。

分年及合并重新计算AUC/AP/IPCW Brier、风险水平、Top20/25覆盖；完整q2函数在所有原生时刻和两个原因上验证。保留全部种子。未运行校准、叶节点收缩、Focal、权重搜索、2025测试或后续调参。公司信息、模型、真实切点、函数参数与预测均不上传。

执行入口（先创建唯一私有运行目录及audit/input/evaluation_input/private/reports/results子目录）：

```bash
python -m experiments.post_st_next_open_info_v1.prepare --run "$RUN"
python -m experiments.post_st_next_open_info_v1.validate_data --run "$RUN"
# 完整测试含独立R通过后，提交代码并冻结
python -m experiments.post_st_next_open_info_v1.stages --stage freeze --run "$RUN" --repo "$REPO"
hetero-submit "$RUN/pilot.jsonl" physical both
python -m experiments.post_st_next_open_info_v1.stages --stage pilot --run "$RUN" --job "$PILOT_JOB"
hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_next_open_info_v1.stages --stage formal --run "$RUN" --job "$FORMAL_JOB"
python -m experiments.post_st_next_open_info_v1.analyze --run "$RUN"
python -m experiments.post_st_next_open_info_v1.audit --run "$RUN"
```

选择现有物理核布局明确使用两台机器；通用hetero-run未校准默认建议SMT，因此不称此布局为实测最快。每任务单线程、4GiB地址空间上限、共享NFS、SHA256验证发布。独立任务可并发，单个森林没有自动跨机器训练。模型拟合与发布成功都须核验；失败不得静默重训。
