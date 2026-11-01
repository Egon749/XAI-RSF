# 全变量删除重训与历史特征选择

保留当前83列q2深度4/共享切点/均衡25/父节点AJ10模型。来源参考提交31cc93d；执行从包含后续历史实验的6b67534开始，不回退或覆盖旧工作。

每个内部入组年度2019–2024，均使用此前训练数据拟合；各外层截止2023/2024/2025只评价截至自己截止前已观察的标签。完整83项删除、相关组、5组随机删9项和固定数量/比例对照均保存真实模型。所有选择只用历史内部验证。

冻结的最低支持条件及删除约束见frozen_protocol.json。支持不足时保留83项，递归路径记录停止，不能把50次回退模型重训中的保留频率当作变量选择可靠性。检查点65/55/45/35不强制达到。

执行顺序：prepare；全测试与协议提交；stages --action freeze；hetero-run pilot.py；逐项哈希gate；inner正式批次；历史汇总与选择决定；bootstrap及outer对照；统一验证、图表及中文报告。每阶段使用独立目录，失败记录保留。

企业明细、切点、模型和完整函数参数仅在私有运行目录。公开代码、汇总CSV、图表、协议、环境和哈希。当前参考指针不变；2023–2025均为已暴露队列，原始披露版本PIT限制继承。

完整复现入口（会执行真实训练；要求仓库已提交且工作区干净，运行路径必须不存在）：

```bash
/home/oliver/.venvs/xai-baselines-20261027/bin/python -m experiments.post_st_dropcolumn_selection_v1.run_pipeline --run /home/oliver/MainDiskData/xai_rsf_dropcolumn_selection_REPRO_UNIQUE
```

该入口保存新私有结果，不覆盖已经发布的报告。失败即停止并保留日志，不自动重跑拟合；仅发布失败应按池回执恢复。核心训练代码的冻结提交、后续分析提交及全部实际作业见execution_receipt.json。
