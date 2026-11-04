# 指定14个不同种子：逐种子32树与448树等权组合

用户原名单含15项，95出现两次。本轮按首次出现顺序去重：98、17、503、5001、2570、22、33、15、99、95、500、2580、97、94。95只训练一次、组合只占一份等权，不重复计数或加倍权重。

复用98/17/503/5001/2570/22/15/99的24个年度森林；99来自更早的12种子实验，额外核对其训练/评价输入哈希及实际模型配置。其余6种子实际拟合18个森林、576棵新树。2024/33完整32树试运行计入18次。

保持74变量、退市支持保护、q≤2、深度4、共享5切点、均衡25变量池、raw直接父节点AJ收缩10、各年度1月1日前信息及24个日历月训练随访上限。三个年度分别等权平均14种子的企业预测，raw/收缩分别合并为448树预测器，不额外拟合或再次收缩。原42/101/202的96树保留。

所有单种子与组合均报告两个原因、raw/收缩版本、年度及合并12/24个月结果。2025年24个月性能NA。1000次原公司级按年度分层配对抽样用于固定预测区间。完整解释核验全部训练/评价行、全部原生时间和两个原因。

```bash
python -m experiments.post_st_requested_seeds14_v1.prepare
# 完整pytest含独立R通过后，提交代码和协议。
python -m experiments.post_st_requested_seeds14_v1.run freeze
python -m experiments.post_st_requested_seeds14_v1.run pilot
python -m experiments.post_st_requested_seeds14_v1.run formal
python -m experiments.post_st_requested_seeds14_v1.run stage_ensemble
python -m experiments.post_st_requested_seeds14_v1.run ensemble
python -m experiments.post_st_requested_seeds14_v1.analyze
python -m experiments.post_st_requested_seeds14_v1.report
```

现有两节点Slurm/NFS调度独立任务，实际节点分配写入回执，不将申请资源等同于实际完成计算。输入与代码冻结，各阶段输出不覆盖历史目录。run_location.json指向唯一私有目录，企业数据、标识、逐企业预测、模型参数和切点留在服务器，公开代码、协议、汇总与哈希。

名单在先前结果已展示后指定，属于探索性比较，不是独立确认。不删除低表现成员、不搜索权重、不按年拼接赢家、不扩展种子搜索，不替换原模型或合并分支。保留全部失败、重试和复用记录。
