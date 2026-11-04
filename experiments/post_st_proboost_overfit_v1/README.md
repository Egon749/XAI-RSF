# PROBoost 训练样本过拟合诊断

[中文结论与图](RESULTS_ZH.md) · [冻结协议](FROZEN_PROTOCOL.json) · [正式模型的训练 AP](ORIGINAL_TRAINING.csv) · [1–48 轮曲线及逐折 AP/AUC](ROUND_CURVES.csv) · [峰值与之后的变化](PEAK_ROUNDS.csv) · [置换分布](PERMUTATION_DISTRIBUTION.csv) · [600 个预设置换的逐次汇总](PERMUTATION_RUNS.csv) · [独立复核与输入哈希](VALIDATION.json) · [回执](execution_receipt.json) · [完整测试日志](pytest.log)

基线为 c349687f9340b5c248e80a6c52b04a7561aad63d。先提交协议，再进行任何诊断。输入仅为原 ESWA 归档三套 74 列训练快照；用于截止日核验的文件是历史拟合元数据。原始公司层级文件、个体得分、抽样细节只在受控的服务器私有目录保存。正式模型的训练 AP 仅通过已保存的 booster 和预处理器计算。

在原研究双节点环境、仓库根目录执行下列命令。ESWA_RUN 指向归档，DIAG_RUN 指向新建的空私有共享目录。对应研究解释器须可访问原 xgboost-cpu 依赖，完整测试须按既有环境设置 CRFOREST_RSCRIPT 及 EW_RLIB。

```bash
export PYTHONPATH="$ESWA_RUN/deps:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
m=experiments.post_st_proboost_overfit_v1
python -m pytest -q "experiments/post_st_proboost_overfit_v1/test_diagnostic.py"
python -m "$m.schedule"
python -m "$m.collect"
python -m "$m.figures"
python -m pytest -q
```

schedule 为每个年度每批 5 个置换生成一项独立作业，真实标签单独成批。collect 校验 123 个发布清单、603 个实验标识和每次标签计数，并核对原始归档模型和训练日志。已有通过校验的计算输出可以直接从 collect 开始核算，不需要重训。分位数采用线性插值，1e-12 用于判定并列和数值满分。cv 是三个折等权指标；oof 使用三折留出公司的合并得分。原始训练 AP 使用 sklearn 独立实现，同时核验每个正负比较对。正式输入、标签和预测均未改写。
