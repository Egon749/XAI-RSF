# 指定16种子：逐种子32树与512树等权组合

固定5001、503、17、2570、5399、39、486、15、8、5100、6270、116、111、98、505、507。前三个种子的三个年度模型核验复用，共9个森林；其余13个种子实际拟合39个森林、1248棵新树。2024/2570完整32树试运行计入39次。每个raw森林派生自身raw直接父节点AJ收缩10版本。

保持74输入、退市支持保护、q≤2、深度4、共享5切点、均衡25变量池、每年1月1日前训练信息、24个日历月训练随访上限。三年度分别等权平均16种子的企业预测，形成512树组合；raw与收缩各自合并，不重训或再次收缩。原42/101/202的96树保留。

全部单种子和组合报告两个原因、raw/收缩、逐年及合并12/24个月结果。2025年24个月性能NA。Top20名额按年ceil(0.2n)，合并指标重新计算。原1000次按年度分层公司抽样用于固定预测的配对区间。完整函数验证覆盖全部训练/评价行、原生时间及两个原因。

```bash
python -m experiments.post_st_requested_seeds16_v1.prepare
# 完整pytest（包括独立R）通过后，提交代码和协议。
python -m experiments.post_st_requested_seeds16_v1.run freeze
python -m experiments.post_st_requested_seeds16_v1.run pilot
python -m experiments.post_st_requested_seeds16_v1.run formal
python -m experiments.post_st_requested_seeds16_v1.run stage_ensemble
python -m experiments.post_st_requested_seeds16_v1.run ensemble
python -m experiments.post_st_requested_seeds16_v1.analyze
python -m experiments.post_st_requested_seeds16_v1.report
```

两节点Slurm/NFS运行独立任务；run_location.json指向唯一私有运行目录。拟合输入和组合输入分别冻结并核验哈希。企业原始数据、标识、模型参数、切点和逐企业预测只留服务器。公开代码、协议、汇总和哈希。全部失败及恢复另列；计算或回传失败不能自动丢弃或重训。

名单在此前结果已展示后指定，属于探索性比较。不同组合的树数也不同，不能把差异只归因于种子。不按年拼接赢家、不搜索组合权重、不扩大种子搜索，不替换原模型或合并分支。
