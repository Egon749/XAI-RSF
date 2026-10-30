# 当前完整模型：结构与幅度恢复模拟

本轮只比较当前 `shallow_parent` 的 q2 和仅解除 q 限制的匹配森林。
83列、每树25列、深度4、共享分位数切点、均衡覆盖、32树/种子、三个种子以及强度10的直接父节点AJ收缩均固定。
当前企业参考模型不变；不读取企业结局、不做新配置搜索。

执行协议见 `frozen_protocol.json`，逐参数差异见 `parameter_differences.csv`。
九种已知CIF场景各50份独立训练样本，主预测器为三个种子的均值。
正式450项独立作业、2700次森林拟合；另有2项数值试运行、4次拟合。
每次拟合保留raw AJ和由其派生的父节点收缩模型；raw仅为次要诊断。

真实函数定义在**已知总体边际乘积参考分布**下。分析导出时重新中心化以对齐真值，预测不变。
相关变量场景须区分乘积参考下函数误差与真实联合分布上的预测误差。
二元输入的投影系数完全刻画相应函数幅度；连续输入同时检查精确L2误差与RMS，不能只用单一系数概括形状。
0.5个百分点RMS为预设检出门槛，0.25/1个百分点为敏感性分析；精确表示不等于作用恢复，更不等于因果识别。

数值验证覆盖全部训练/5000个测试输入在365/730天的预测、32个固定探针在所有原生时间点的预测、4个探针的完整逐项加和。
每个叶节点在所有原生时间点检查CIF合法性，因此正权平均后的森林也合法。绝不裁剪预测。
不限定q的模型使用包含实际三/四阶项的完整表示。缺失状态计入逻辑表格单元，但本轮DGP没有缺失值。

执行入口（运行目录必须为新的唯一私有目录）：

```bash
python -m experiments.post_st_structure_recovery_v1.prepare --stage freeze --run PRIVATE_RUN
/home/oliver/.local/bin/hetero-run PRIVATE_RUN/pilot.py
python -m experiments.post_st_structure_recovery_v1.prepare --stage pilot_gate --run PRIVATE_RUN --job JOB --report REPORT
/home/oliver/.local/bin/hetero-run PRIVATE_RUN/formal.py
python -m experiments.post_st_structure_recovery_v1.prepare --stage formal_gate --run PRIVATE_RUN --job JOB --report REPORT
python -m experiments.post_st_structure_recovery_v1.analyze --run PRIVATE_RUN
```

私有输入为已提交代码与冻结协议；随机种子决定模拟数据，输入哈希逐任务保存。
两机使用已有Slurm池和NFS；每任务单线程，六次配对拟合顺序执行，任务之间并行。
模型原始JSON采用无损gzip封装并重载核对；模型及逐模拟数据保留在私有目录，公开汇总诊断。

试运行前检查记录：新增测试第一次运行11通过、1失败，原因为测试误写信号行数12；正确数量是四个真实非基线函数×两个时点=8。修正该计数断言后全部通过，未改数值容差、数据、模型或删减检查。
