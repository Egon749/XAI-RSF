# C039 概率水平修复探索

本轮复用当前74输入、17/503/22、96树、q2深度4、退市支持保护、直接父节点AJ10主模型。固定7臂，每年度独立计算。没有重训森林或替换研究参考。

优先读 [RESULTS_ZH.md](RESULTS_ZH.md)、[FROZEN_PROTOCOL.json](FROZEN_PROTOCOL.json)、[execution_receipt.json](execution_receipt.json)。

## 入口和复现

1. 在自己的新分支提交冻结协议。本轮冻结提交45cf4b0e42f59cd45cb71f4ccda65d135b8275a7。
2. 在原私有父目录下用 tempfile.mkdtemp 创建新的唯一目录，设置 REPAIR_RUN。不要复用已完成目录运行 prepare/schedule。
3. 激活已有Python环境 /home/oliver/.venvs/xai-baselines-20261027/bin/python。
4. 执行：

~~~bash
python -m experiments.post_st_probability_repair_v1.prepare
python -m experiments.post_st_probability_repair_v1.schedule
python -m experiments.post_st_probability_repair_v1.evaluate
python -m experiments.post_st_probability_repair_v1.report
~~~

schedule使用已提交的crforest与experiments代码快照、3个独立年度任务及现有hetero-run/NFS。worker只见训练标签和评价X，预测写完后才由evaluate读取评价结局。跨年历史校准标签来自各自截止日前快照，不回填较晚结局。

本次私有位置见run_location.json。真实公司、输入、模型、函数表、逐公司预测、映射参数和未抑制分组统计均在私有运行目录；公开CSV只有汇总。旧目录未改写。

## 输出

- DATA_AUDIT.json：输入/原模型哈希、训练与校准支持、参考预测恒等。
- METRICS_ALL.csv：所有臂、年度/pooled、两原因、12/24个月；2025H24预测均值保留，性能NA。
- PAIRED_DIFFERENCES.csv：映射减原模型的1000公司配对条件区间，undefined计数。
- JOINT_BRIER.csv：两原因Brier之和，不额外加无事件状态项。
- RISK_GROUP_CALIBRATION.csv：固定原排序Top20及剩余组；小事件与互补小格已抑制。
- FUNCTION_VALIDATION.csv：21组全样本全时间完整函数/概率/重载验证、函数数及导出资源。
- MAPPING_DIAGNOSTICS.csv：数值求解及证书诊断；拟合系数仅私有保存。
- RANK_IDENTITY.csv：逐年并列/排序恒等检查。
- EVALUATION_AUDIT.json：原评价复现、输入对齐及共同重抽样哈希。
- figures/：年度预测与观察图，PNG/PDF。
- FAILURES.json：实验臂失败清单。本轮为空；环境和汇总脚本错误单列于回执。

全部结论为探索性，历史原始财务版本/PIT限制与C039种子后选的乐观偏差均继承。无自动采用门槛，无新模型替换。

## 完整测试环境

~~~bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CRFOREST_RSCRIPT=/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript
export EW_RLIB=/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib
export PYTHONPATH=/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/drop9_support_20261102_p4gfo7x4/private/eswa_baselines_v1_wMGO11Fz/deps:$PWD
/home/oliver/.venvs/xai-baselines-20261027/bin/python -m pytest -q
~~~

结果339 passed，1个既有EBM可视化警告。日志见pytest.log；独立R检验包括在完整测试中。
