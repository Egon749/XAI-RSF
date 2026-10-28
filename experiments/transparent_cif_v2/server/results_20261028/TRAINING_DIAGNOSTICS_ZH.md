# 实际训练与数值诊断

权威提交fd4a87aee54c70d5e4d165b1e7fa7011fc43c5db；正式训练冻结提交6a4a30ef16f3ffeb9f0d5636a065dde5996ce8d2。核心森林、分区、函数编译、成对候选与权重算法未改数学实现。本次增加服务器适配/独立核验和真实InterpretML测试；合成demo仅补记录主效应幅度与oracle收据。

旧＋新完整测试：原权威版本98 passed；补真实InterpretML/Parquet端到端测试后99 passed，0 failed、0 skipped。包含R survival独立AJ核验和randomForestSRC单分裂/叶AJ核验、Parquet与真实InterpretML 0.7.8。不能将这两项R检查理解成完整森林与randomForestSRC逐项等价。唯一警告为EBM默认图不显示缺失；数值缺失单元保留，不填极值、不裁剪预测。

固定42/101/202三种子，六匹配方案：base_equal、base_weighted、ordinary_extra_equal、ordinary_extra_weighted、pair_equal、pair_weighted。每种子1个32树基础拟合、增加16棵普通树及16棵联合候选树，形成3个树库，分别等权/学习权重。18个部署模型不是18个完全独立基础森林拟合。每种子2种EBM交互预算×3种轮数，共18次EBM候选拟合，保留6个选中模型；全部选择50轮，只用历史内部公司留出。全部轮数分数见ebm_selection.csv。

正式Slurm作业2546的4项任务全部计算成功且哈希发布成功，包含三种子train→predict与一个固定模拟任务。调度使用双节点池physical模式，实际这4项均被oliver领取；不冒称两机同时训练。bootstrap作业2547采用p模式，oliver的8个P物理核与lemon的物理核处理10批任务，实际8批oliver、2批lemon；没有使用P核第二线程或绕过cgroup。单任务BLAS1，正式训练8GiB地址空间上限，非GPU算法。

| seed | training_seconds | seconds | peak_children_rss_MiB | host |
| --- | --- | --- | --- | --- |
| 42 | 85.9986 | 98.3177 | 533.281 | oliverdesktop |
| 101 | 84.8614 | 97.0514 | 533.723 | oliverdesktop |
| 202 | 83.377 | 95.7508 | 533.723 | oliverdesktop |

候选与支持：

| seed | original_trees | added_trees | candidate_pairs | evaluated_partitions | supported_partitions | selected_pairs | no_supported_candidate_pairs | selected_score_min | selected_score_max | candidate_pairs_present_before | candidate_pairs_present_after | actual_pairs_before | actual_pairs_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 32 | 16 | 64 | 1782 | 1278 | 16 | 1 | 823.652 | 1366.85 | 32 | 32 | 42 | 42 |
| 101 | 32 | 16 | 64 | 1634 | 1118 | 16 | 3 | 978.8 | 1449.59 | 32 | 34 | 36 | 38 |
| 202 | 32 | 16 | 64 | 1662 | 1122 | 16 | 5 | 837.071 | 1140.74 | 33 | 34 | 38 | 39 |

每种子64个候选变量对、最多3个切点、最多16棵新树。联合评分四区域，每格必须满足原记录/公司/事件支持阈值；评分不是统计显著性。评估分区5078个，其中3518个支持合格。64是候选审计清单长度，不是实际森林交互数；实际交互42→42、36→38、38→39。原始逐候选日志含所有失败支持和分数，保留私有。一次汇总初稿误把coverage列表当计数，已纠正并保留原稿及diagnostic_correction.json；训练与预测没有改变。

| seed | bank | status | iterations | cleanup_iterations | frank_wolfe_gap | objective | uniform_objective | effective_tree_count | positive_weights | largest_weight | simplex_roundoff_adjustment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | base | CONVERGED | 65 | 24 | 9.60335e-09 | 0.0279853 | 0.028508 | 12.554 | 22 | 0.134777 | 0 |
| 42 | ordinary_extra | CONVERGED | 70 | 16 | 8.7855e-09 | 0.0279334 | 0.0286185 | 11.7956 | 37 | 0.125858 | 0 |
| 42 | pair | CONVERGED | 67 | 38 | 9.86485e-09 | 0.0279907 | 0.0287855 | 13.0802 | 36 | 0.131331 | 0 |
| 101 | base | CONVERGED | 68 | 6 | 7.80998e-09 | 0.0280505 | 0.0285533 | 9.73388 | 19 | 0.184244 | 0 |
| 101 | ordinary_extra | CONVERGED | 83 | 8 | 9.5339e-09 | 0.0279572 | 0.0284849 | 10.7331 | 34 | 0.161311 | 0 |
| 101 | pair | CONVERGED | 56 | 41 | 9.3556e-09 | 0.0280385 | 0.0287548 | 10.2497 | 30 | 0.172869 | 0 |
| 202 | base | CONVERGED | 61 | 52 | 9.85229e-09 | 0.0279372 | 0.0285568 | 8.47195 | 23 | 0.206211 | 0 |
| 202 | ordinary_extra | CONVERGED | 89 | 8 | 9.62291e-09 | 0.0278362 | 0.028417 | 10.4239 | 37 | 0.176161 | 2.77556e-17 |
| 202 | pair | CONVERGED | 75 | 50 | 9.52903e-09 | 0.0279473 | 0.0288127 | 8.62203 | 28 | 0.205209 | 0 |

9次QP均收敛且最终gap≤1e-8，没有未收敛后继续训练。正权重数与有效树数均记录；其中一次数值单纯形修正2.78e-17，原收据保留。这是参数浮点可行性修正，不是对预测裁剪。每企业、原因、时间使用同一套非负和为1的树权重，没有残差或门控。

损失是历史内部留出IPCW的K+1状态Brier＋0.001趋近等权正则；内部G最小约0.165，且竞争事件仅5个独立事件。内损失改善不等于2023/2024性能改善，也不能据本次结果认定稀少竞争事件是唯一原因。

实际任务失败0；没有选择性删种子、调预算、重训M2或追加搜索。报告修正与既有容器缺依赖记录分开保留。独立部署18/18成功，全体23799记录的两个原因CIF与原始加权树逐一核验；完整全时间格的概率/单调性额外检查64个固定记录与全缺失输入。不是把65条全时间检查冒充全部23799条全时间实测。
