# 每种子32树：20个随机种子测试

固定74变量、q≤2、深度4、退市支持保护、共享切点、均衡25变量池、直接父节点AJ收缩10、24个日历月训练随访。每个年度均在1月1日信息截止前的冻结数据上训练。

20个种子固定为7、17、23、42、59、73、89、101、131、167、202、251、313、419、503、619、733、887、1009、2026。每种子32树，分别报告；不合并20个种子，不从暴露结果选择新种子。原42/101/202各32树平均的96树模型独立保留。

每年20个模型，三年60个：9个已有模型原样复用，51个新拟合。2024/seed7的完整32树数值试运行成功后作为该格正式结果，不重复拟合；剩余59项任务含50个新拟合和9个复用导出。新拟合共1632棵树，无校准。

此前未执行的“每种子96树”准备仅是待用户确认的草案，已在其私有运行目录归档；用户明确32树后新建了独立运行目录，原草案没有拟合或Slurm作业。

```bash
python -m experiments.post_st_seed_stability20_v1.prepare
# 完整pytest（含独立R），提交协议和代码后：
python -m experiments.post_st_seed_stability20_v1.run freeze
python -m experiments.post_st_seed_stability20_v1.run pilot
python -m experiments.post_st_seed_stability20_v1.run formal
python -m experiments.post_st_seed_stability20_v1.analyze audit
python -m experiments.post_st_seed_stability20_v1.analyze metrics
python -m experiments.post_st_seed_stability20_v1.analyze functions
python -m experiments.post_st_seed_stability20_v1.report
```

run_location.json记录独立私有目录；数据、公司标识、预测、模型和函数参数只保留私有。公开为代码、配置、聚合指标和哈希。2023–2025均已暴露，是随机性探索，不是20个独立验证队列；2025年24个月性能NA。保留每个种子的退市/恢复、12/24月、raw/收缩结果，主表展示收缩版本。汇总均值指20个指标的均值，不是20份预测平均后的新模型指标。
