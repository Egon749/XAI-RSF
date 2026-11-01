# 6月30日训练截点与后续12个月入组窗口

用户明确选择把2023、2024、2025年6月30日作为**训练信息截止日（含当日）**。这不是统一企业预测原点，也不是只评价上半年入组企业。每个冻结模型用于随后12个日历月内新入组企业；各企业仍从其ST公告后的下一交易日开盘前，预测自身未来12个日历月先最终退市/先恢复。

比较原83变量与固定删除九项后的74变量。q2、深度4、每树25变量均衡覆盖、共享切点、32树及种子42/101/202、直接raw父节点AJ收缩10保持不变。训练随访上限仍24个月，按各截点重新截断，而评价时点为12个月。不新增筛选、校准或搜索，不更换当前模型。

共18次新森林拟合，全部保留raw与收缩版本。训练数据和测试标签隔离。单种子及三种子平均完整函数在全部训练/评价记录、原生时间点和两原因验证，容差1e-10。

2025截点计划评价的入组窗口为2025-07-01至2026-06-30，但当前审核冻结队列只覆盖至2025年；窗口内仅有1家已冻结企业。保存模型和私有预测，公开评价指标全部留空，不能把这1家当作完整未来队列。

## 复现入口

在仓库根目录，用既有研究Python环境执行；RUN必须是全新的私有运行目录。

```bash
python -m experiments.post_st_june_cutoff_v1.prepare --stage prepare --run "$RUN"
# 完整pytest通过，并将代码/协议提交后才能冻结
python -m experiments.post_st_june_cutoff_v1.prepare --stage freeze --run "$RUN"
/home/oliver/.local/bin/hetero-run "$RUN/formal.py"
python -m experiments.post_st_ablation_v1.stages --stage formal --run "$RUN" --job "$JOB" --report "$REPORT"
python -m experiments.post_st_june_cutoff_v1.analyze --run "$RUN"
python -m experiments.post_st_june_cutoff_v1.deliver --run "$RUN"
```

资源为两节点Slurm池、18个独立单线程任务、每任务4GiB地址空间上限；不重复整份实验充当并行。池的实际主机分配以回执为准。企业记录、真实预测、函数参数和切点仅保留服务器私有目录。
