# 指定11种子：逐种子32树与352树等权组合

固定种子17、1009、503、101、3003、3509、2903、2560、3721、4590、5001。前4个×3年度复用12个已核验森林；后7个×3年度实际拟合21个森林、672棵新树。2024/3003完整32树试运行计入21次。每个raw森林派生自身raw直接父节点AJ收缩10版本。三个年度分别合并11个raw/收缩成员为352树预测器；为并行加速，raw/收缩分成6个独立导出任务，不重训、不再次收缩。

仍为74输入、q≤2、深度4、共享5切点、均衡25变量池、退市支持保护、各年1月1日前信息与24个日历月训练随访。原42/101/202组合96树保留。每个种子和组合均评价两个原因、raw/收缩、12/24个月及合并年度；2025年24个月性能NA。

```bash
python -m experiments.post_st_requested_seeds11_v1.prepare
# 完整pytest含独立R通过后提交代码与协议。
python -m experiments.post_st_requested_seeds11_v1.run freeze
python -m experiments.post_st_requested_seeds11_v1.run pilot
python -m experiments.post_st_requested_seeds11_v1.run formal
python -m experiments.post_st_requested_seeds11_v1.run stage_ensemble
python -m experiments.post_st_requested_seeds11_v1.run ensemble
python -m experiments.post_st_requested_seeds11_v1.analyze
python -m experiments.post_st_requested_seeds11_v1.report
```

使用现有两节点Slurm/NFS。拟合和组合输入分别冻结，核对所有源和输出哈希。run_location.json指向唯一私有目录。公司标识、原始输入、逐公司预测、模型、真实切点与完整函数参数只留服务器，公开代码、协议和汇总。复用不计作新拟合，公开计数分别列出。

名单在已查看年度种子排名后由用户指定，本轮属于探索性比较，不是无偏种子抽样或独立确认。完整精确解释和概率合法性仍须逐项验证。保留不利结果与失败记录；不按年拼接最优种子、不优化组合权重、不自动更换原96树/current pointer/主模型，不合并分支。
