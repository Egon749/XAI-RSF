# 指定16种子第二轮：每种子32树与512树等权组合

固定505、98、17、503、5001、2570、15、19、20、5399、22、5000、4999、4998、2500、100。复用505/98/17/503/5001/2570/15/5399的24个年度森林；8个新种子实际拟合24个森林、768棵新树。2024/19的完整32树试运行计入24次。

保持74变量、退市支持保护、q≤2、深度4、共享5切点、均衡25变量池、raw直接父节点AJ收缩10、各年度1月1日前信息与24个日历月训练随访上限。三年度分别合并16种子预测，raw/收缩各自形成512树预测器，不重训、不重复收缩。原42/101/202的96树保留。

评价所有单种子与组合、两个原因、raw/收缩版本、年度及合并12/24个月结果；2025年24个月性能NA。1000次原公司级按年度分层配对抽样，固定预测。完整解释核验全部训练/评价行、全部原生时间、两个原因。所有模型、参数、企业数据与预测留在run_location.json所指唯一私有目录；公开代码、协议、汇总和哈希。

```bash
python -m experiments.post_st_requested_seeds16_v2.prepare
# 完整pytest含独立R通过后，提交代码与协议。
python -m experiments.post_st_requested_seeds16_v2.run freeze
python -m experiments.post_st_requested_seeds16_v2.run pilot
python -m experiments.post_st_requested_seeds16_v2.run formal
python -m experiments.post_st_requested_seeds16_v2.run stage_ensemble
python -m experiments.post_st_requested_seeds16_v2.run ensemble
python -m experiments.post_st_requested_seeds16_v2.analyze
python -m experiments.post_st_requested_seeds16_v2.report
```

现有两节点Slurm/NFS运行独立任务。预检lemon约30GiB内存可用，拟合峰值历史小于300MiB，因此本轮内存并发上限采用oliver24、lemon16；P/E核和SMT亲和性由Slurm管理，模型数学与随机流不变。单进程不跨机器加速。

名单在查看过既有结果后指定，属于探索性比较。全部成员预先固定，不丢弃低表现种子，不搜索权重，不按年拼接赢家，不替换当前模型或合并分支。不同组合名单与树预算可能同时变化，不将差异只归于某一项。失败、重试和复用均单列。
