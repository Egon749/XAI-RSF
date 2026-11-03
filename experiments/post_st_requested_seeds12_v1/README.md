# 指定12种子：逐种子32树及等权384树组合

固定种子17、1009、503、42、101、99、1000、889、24、555、499、1010。前5个种子×3年度复用15个已核验森林；新增7个×3年度实际拟合21个森林、672棵新树。2024/99完整32树试运行计入21次拟合。每个raw森林仅派生自身直接父节点AJ收缩10版本。最后分别合并12个raw/收缩成员为384树预测器，不重训、不再收缩。

仍为74输入、q≤2、深度4、共享5切点、均衡25变量池、退市支持保护、各年1月1日前信息和24个日历月训练随访。原42/101/202组合96树保留。逐种子与组合都评价两个原因、raw/收缩、12/24月及合并年度，2025年24月性能NA。

```bash
python -m experiments.post_st_requested_seeds12_v1.prepare
# 全部pytest含独立R通过，提交代码与协议。
python -m experiments.post_st_requested_seeds12_v1.run freeze
python -m experiments.post_st_requested_seeds12_v1.run pilot
python -m experiments.post_st_requested_seeds12_v1.run formal
python -m experiments.post_st_requested_seeds12_v1.run stage_ensemble
python -m experiments.post_st_requested_seeds12_v1.run ensemble
python -m experiments.post_st_requested_seeds12_v1.analyze
python -m experiments.post_st_requested_seeds12_v1.report
```

独立任务采用现有两节点Slurm/NFS，拟合与组合输入分别冻结和核对哈希。run_location.json指向唯一私有运行目录。所有企业数据、模型、真实阈值、逐公司预测和函数参数留服务器；公开汇总、代码、配置、哈希。复用不计为新拟合。

名单由用户在查看历史种子排名后指定，是探索性指定种子实验，不是无偏随机种子分布或独立确认。保留所有不利结果及失败记录，不自动替换原96树、主模型、CURRENT_POST_ST_MODEL.json或修改核心算法，不合并分支。所有数据/PIT与74变量选择限制沿用。
