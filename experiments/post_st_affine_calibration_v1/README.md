# ST后12个月仿射再校准：2023／2024时间外实验

三段独立公司集：先训练森林，再用随后两年校准，最后评价下一年。两阶段均按当时可得结局截断，未知结果保留删失。

固定比较 q2、同后端无限阶和标准RSF，三个种子；主要比较先平均三个种子的预测再拟合一次校准器，不混淆“各自校准后平均”与仿射排序保持。没有规则特征或原因权重变化。观察期沿用12个月。

仿射截距曲线采用训练AJ曲线归一化，365天后保持常数；这保证365／366天日历年终点采用同一仿射映射。仅在12个月拟合校准，不宣称中间时间也已校准。采用整个概率单纯形的充分约束，避免只在观察样本上有效。

协议见 frozen_protocol.json。2023校准集仅1例退市，结果必须报告支持不足。基础训练集为校准独立性缩小，不能直接把本轮未校准指标与旧全样本森林当作算法差异。企业和参数留在私有目录。

## 实际结果

18个森林、24个校准器、2个试运行完成，91项测试通过。q2集成退市Brier：2023为0.109322→0.106667；2024为0.319896→0.294980；AUC/AP、Top20/25名单不变。最大函数重构误差1.67e-16。2023退市输出接近常数，2024平均风险仍明显低估。判断INCONCLUSIVE。

- [完整结果](RESULTS_ZH.md)、[事件核查](PREDICTION_COMPARISON_ZH.md)、[解释与概率约束](INTERPRETABILITY_COMPARISON_ZH.md)。
- [数据审计](DATA_ALIGNMENT_AUDIT_ZH.md)、[数值与执行诊断](TRAINING_DIAGNOSTICS_ZH.md)、[研究判断](METHOD_INNOVATION_DECISION_ZH.md)。

## 复现入口

创建唯一私有RUN及 input/audit/reports/private/results/evaluation_input 空目录；不要指向既有运行。使用已安装的隔离环境，先运行完整pytest（含独立R检查）。

```bash
python -m experiments.post_st_affine_calibration_v1.prepare --run "$RUN"
python -m experiments.post_st_affine_calibration_v1.stages --run "$RUN" --repo "$PWD" --stage freeze
/home/oliver/.local/bin/hetero-submit "$RUN/pilot.jsonl" physical both
python -m experiments.post_st_affine_calibration_v1.stages --run "$RUN" --stage pilot --job "$PILOT_JOB"
/home/oliver/.local/bin/hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_affine_calibration_v1.stages --run "$RUN" --stage formal --job "$FORMAL_JOB"
python -m experiments.post_st_affine_calibration_v1.analyze --run "$RUN" --stage calibrate
python -m experiments.post_st_affine_calibration_v1.analyze --run "$RUN" --stage evaluate
python -m experiments.post_st_affine_calibration_v1.validate --run "$RUN"
```

训练前必须冻结代码；只在试运行和正式阶段门检通过后进入下一步。所有输出唯一创建，禁止覆盖历史目录。环境版本、模型及输入哈希见results与execution_receipt.json。企业明细、校准系数、预测及原函数参数留在服务器。
