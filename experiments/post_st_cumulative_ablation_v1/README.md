# 固定顺序累积消融

完整q2模型 → 解除q限制 → 再改局部随机切点 → 再改IID25 → 再取消父节点收缩。保留深度4、每树25变量、种子42/101/202、32树及全部其他设置。前两阶段复用12个模型；中间两个新结构各拟合6次；末阶段使用同一s3森林的raw AJ，不重复拟合。

这条链只估计给定删除顺序下的条件变化，不是顺序无关的独立贡献。s4也不是randomForestSRC标准RSF。既有单项消融与当前研究参考不变。

已完成12次新增拟合、12个收缩派生版本和40项完整验证；完整测试155通过、0跳过。结果、限制与全部反向发现见[中文结果](RESULTS_ZH.md)、[预测与校准](PREDICTION_COMPARISON_ZH.md)、[复杂度与稳定性](COMPLEXITY_COMPARISON_ZH.md)、[执行诊断](TRAINING_DIAGNOSTICS_ZH.md)、[数据审计](DATA_ALIGNMENT_AUDIT_ZH.md)、[论文方法与结果段落](PAPER_METHODS_RESULTS_ZH.md)和[执行回执](execution_receipt.json)。图表和可复核CSV在results/，模型及公司明细只在回执指向的私有服务器目录。

使用已存在的私有旧C工件，在新的空RUN目录按下列顺序执行。freeze要求通过完整测试并提交后的干净工作区；调度器返回真实JOB和REPORT路径后再执行gate。所有阶段排他创建，不覆盖历史结果。

```bash
python -m experiments.post_st_cumulative_ablation_v1.prepare --stage prepare --run "$RUN"
python -m experiments.post_st_cumulative_ablation_v1.prepare --stage freeze --run "$RUN"
hetero-run "$RUN/formal.py"
python -m experiments.post_st_ablation_v1.stages --stage formal --run "$RUN" --job "$FIT_JOB" --report "$FIT_REPORT"
python -m experiments.post_st_cumulative_ablation_v1.prepare --stage verify_manifest --run "$RUN"
hetero-run "$RUN/verify.py"
python -m experiments.post_st_ablation_v1.stages --stage verify --run "$RUN" --job "$VERIFY_JOB" --report "$VERIFY_REPORT"
python -m experiments.post_st_cumulative_ablation_v1.analyze --run "$RUN"
python -m experiments.post_st_cumulative_ablation_v1.report --run "$RUN"
```

调用原消融已验证的模型保存、完整高阶导出、逐节点AJ/分裂重放和调度gate，不修改核心森林或旧分析代码。通过独立只读工件清单适配新的五阶段名称。DATA_DIR仅含训练数据和预测X/ID，测试结局单独保存；单线程独立作业使用两机Slurm和NFS，峰值资源和发布状态逐项记录。企业数据、预测、真实切点、模型及完整函数始终留服务器私有目录。
