# ST后预测的原始数据核查：未启动新训练

直接核查原始事件公告、财报发布日期、利润表字段与日行情后，发现原来的75列输入尚未充分适配“已被ST后的恢复／退市”任务：ST与*ST、审计意见和持续经营信息没有入模，严格早于入组日的财报筛选又排除了大量同日年报。

不过，测试期ST与*ST的退市比例差异有限；股价本来就在输入中。不能据此断言加上几个制度变量就能修复模型。

- [原数据分析及结论](RAW_DATA_ANALYSIS_ZH.md)：完整数量、退市依据、信息日期和可恢复字段。
- [汇总CSV](results/)：原始数据核对、类型分布、年度报告信息缺口和单变量排序诊断。
- [执行回执](execution_receipt.json)：源文件、旧输入与新增产物哈希及测试结果。
- [原模型结果](../post_st_prognosis_v1/RESULTS_ZH.md)：保持原负面结果和原模型不变。

范围仍是全部非金融行业、首次财务／审计类ST或*ST；训练309家，2023年36家、2024年45家，24个日历月内首次完全恢复与最终退市竞争。公告日为时间依据。此轮为接触结局后的探索性数据审计，数据仍为 **PROVISIONAL**。

本轮新拟合数 **0**。仅在独立私有目录准备了增加 `entry_is_star_st` 的76列候选输入，未用于拟合、预测或替换原75列输入。原始公告文字、企业行、模型、预测、具体市值和财务数据不提交Git。

复现时使用新的私有运行目录，两个入口都拒绝覆盖已有产物：

```bash
python -m experiments.post_st_rule_features_v1.prepare \
  --base-run /path/to/immutable/post_st_run \
  --run /path/to/new/private/audit_run \
  --original-receipt experiments/post_st_prognosis_v1/execution_receipt.json
python -m experiments.post_st_rule_features_v1.inspect_raw \
  --base-run /path/to/immutable/post_st_run \
  --run /path/to/new/private/audit_run
```

入口读取服务器现有权威数据路径，不下载替代样本。`terminal_ground` 是针对当前45个已观察退市案例、经原文核查的描述性分类，不是通用法律判定器；终局公告仅用于结局分析，绝不回填为入组特征。
