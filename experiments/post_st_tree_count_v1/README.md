# 树数有限对照

固定当前74变量、退市支持保护、q2、深度4、共享切点、均衡25变量、直接父节点AJ收缩10和24个月训练随访，只改变每个种子树数8/16/32/64/128/256。三个种子42/101/202平均，最终预测器对应24/48/96/192/384/768棵树。原模型和主模型文件夹保留，不自动采用任何新树数。

先用合成测试证明分块树编号、前缀、重新训练与父节点收缩的一致性，再复用现有9个32树raw森林，新增2016棵树，组成9个256树森林。较小候选取固定编号前缀，属于相同随机规则的精确模型，不是删除函数项。每树bootstrap和均衡覆盖均由原seed/tree_id生成；无全局单元预算，树之间没有顺序依赖。新训练分63个独立32树任务，两节点Slurm调度；另2树试运行不用于选择。

复现必须使用新的唯一私有运行目录；不能直接向已完成目录重跑。整个仓库的既有代码与私有输入必须可用。

```bash
python -m experiments.post_st_tree_count_v1.prepare
# 完整pytest（包括独立R），提交冻结协议和运行代码
python -m experiments.post_st_tree_count_v1.run freeze
python -m experiments.post_st_tree_count_v1.run pilot
python -m experiments.post_st_tree_count_v1.run grow
python -m experiments.post_st_tree_count_v1.run assemble
python -m experiments.post_st_tree_count_v1.run export_pilot
python -m experiments.post_st_tree_count_v1.run inspect
python -m experiments.post_st_tree_count_v1.analyze metrics
python -m experiments.post_st_tree_count_v1.analyze functions
```

2023–2025已暴露，本轮仅能报告有限网格中的探索性表现、随机波动、复杂度和代价，不能宣称确认了普遍最优树数。只使用各年Jan1之前的训练快照，入组时点特征不更新，未完成随访不当负例；2025年24个月性能NA。原74变量名单和披露版本限制继承。主要12个月退市；两个原因、raw/收缩、各种子、年度/pooled/年内AUC和支持的24个月均保留。

函数导出在全部训练/评价样本、全部原生时间和两个原因核验≤1e-10，不裁剪。1000次公司级配对区间只描述固定模型测试样本波动，不包含树数选择与完整训练不确定性。公司数据、阈值、模型和函数表保存在run_location.json所指私有目录，只发布汇总。
