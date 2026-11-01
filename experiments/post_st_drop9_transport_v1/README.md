# 固定删除九项变量：2023与2025匹配实验

沿用post_st_drop9_v1的同一九项名单，两年各新增三个32树森林，保留原83列模型。q2、深度4、共享切点、均衡25变量、mtry75、叶设置10/5/3、原因权重(1,1)、单位时间权重及直接父节点AJ收缩10全部不变；保留raw与shrunk文件。

2023沿用2012—2022训练和2023-01-01前信息，评价12/24个月。2025沿用2012—2024训练和2025-01-01前信息，观察截止2026-06-30；只评价有随访支持的12个月，24个月保留预测与支持度，不制造终点指标。全部年度先平均三个种子的公司预测，再评价。

原83列参考直接复用；公司bootstrap配对，输入保留列逐值一致，切点字典一致。维数减少使均衡覆盖周期与具体变量资格集合改变，这一变化包含在删变量策略中。名单来源与用户选定同九项的决定记入冻结协议。

运行顺序：

```bash
python -m experiments.post_st_drop9_transport_v1.prepare --stage prepare --run RUN
# 完整测试通过并提交冻结代码后
python -m experiments.post_st_drop9_transport_v1.prepare --stage freeze --run RUN
/home/oliver/.local/bin/hetero-run RUN/formal.py
python -m experiments.post_st_ablation_v1.stages --stage formal --run RUN --job JOB --report REPORT
python -m experiments.post_st_drop9_transport_v1.analyze --run RUN
```

RUN必须是新建空私有目录。模型、逐公司预测、切点和函数参数留服务器，仓库保存汇总结果、代码、配置、环境及哈希。当前参考模型不自动替换，无追加参数搜索。
