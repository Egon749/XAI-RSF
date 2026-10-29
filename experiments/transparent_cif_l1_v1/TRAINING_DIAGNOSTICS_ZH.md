# 执行、收敛与故障账本

实际3个seed主运行；3次基础森林拟合=96棵基础树；普通新增48棵，入选成对48棵。成对搜索共192候选对、5038次分区评价，182棵支持合格候选树被构造后按既定上限选择。共9次权重求解、18份单seed透明预测器及6份无重训集成。不是18次独立森林重训。

真实EBM候选18次、选中6个，全部选50轮；没有内部留出再拟合。固定demo另有3基础森林、9标签权重和9真值容量求解，27比较全部保留。单元测试合成fit、子进程CLI fit、R和真实InterpretML计数见execution_receipt.json；初始177项、最终183项全部通过，无skip，均包含独立R/Parquet/真实InterpretML。

| stage | tasks | seconds | hosts |
| --- | --- | --- | --- |
| demo | 1 | 2.52686 | {'oliverdesktop': 1} |
| train_oliver | 2 | 86.3239 | {'oliverdesktop': 2} |
| train_lemon | 1 | 124.59 | {'lemon': 1} |
| predict | 3 | 28.8165 | {'lemon': 3} |
| native_audit | 24 | 187.945 | {'oliverdesktop': 24} |
| bootstrap_recovery | 10 | 1.09273 | {'oliverdesktop': 10} |


seed42/101在oliver，202在lemon，单seed串行1CPU/BLAS1，未声称单进程跨机或GPU参与。两节点原隔离环境未修改。lemon申请1物理CPU/8GiB，原pool负责oliver affinity。完整原生核验/独立bootstrap批次使用两节点池。各阶段时间是实际阶段墙钟，不等于总端到端或可相加的拟合CPU。原bootstrap数值阶段墙钟82.744秒，另列恢复发布时间，不能把恢复耗时当作全部bootstrap耗时。逐任务耗时与峰值RSS见task_costs.csv。

| seed | candidate_pairs | evaluated_partitions | supported_partitions | unsupported_pairs | selected_pairs | pair_search_seconds | ordinary_extension_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 64 | 1782 | 1278 | 1 | 16 | 7.41642 | 21.087 |
| 101 | 64 | 1634 | 1118 | 3 | 16 | 6.69884 | 20.5263 |
| 202 | 64 | 1622 | 1109 | 6 | 16 | 9.80678 | 29.1175 |


普通和成对总树数都48，但搜索方式/成本不同；不称相同算力预算。失败支持的候选没有通过放宽门槛补足。

| seed | bank | status | iterations | cleanup_iterations | frank_wolfe_gap | objective | uniform_objective | effective_tree_count | largest_weight | constraint_residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | base | CONVERGED | 48 | 43 | 9.35006e-09 | 0.0280044 | 0.0285318 | 12.5085 | 0.134884 | 0 |
| 42 | ordinary_extra | CONVERGED | 73 | 30 | 8.55329e-09 | 0.0279547 | 0.0286426 | 11.8517 | 0.126616 | 1.11022e-16 |
| 42 | pair | CONVERGED | 61 | 36 | 9.60679e-09 | 0.0280091 | 0.0288104 | 13.0167 | 0.131229 | 2.22045e-16 |
| 101 | base | CONVERGED | 62 | 7 | 9.63767e-09 | 0.028075 | 0.0286 | 9.89369 | 0.187327 | 1.11022e-16 |
| 101 | ordinary_extra | CONVERGED | 77 | 14 | 9.85098e-09 | 0.0279826 | 0.0285229 | 10.8208 | 0.164294 | 0 |
| 101 | pair | CONVERGED | 66 | 24 | 7.48689e-09 | 0.0280612 | 0.0287961 | 10.206 | 0.176021 | 2.22045e-16 |
| 202 | base | CONVERGED | 64 | 45 | 9.96284e-09 | 0.0279648 | 0.0285929 | 8.45098 | 0.206591 | 0 |
| 202 | ordinary_extra | CONVERGED | 86 | 23 | 9.77932e-09 | 0.027862 | 0.0284507 | 10.4286 | 0.177182 | 0 |
| 202 | pair | CONVERGED | 73 | 20 | 7.77776e-09 | 0.0279753 | 0.02884 | 8.45098 | 0.206592 | 2.22045e-16 |


9/9收敛，最大FW gap 9.96284e-09≤1e-8。单seed有效树数约8.45—13.02，最大权重约0.207：有集中，但没有单树独占；内优化收敛不保证后期泛化。原显式浮点单纯形修复保留于收据，不裁剪预测。

故障完整保留：lemon-only快捷包装器拒绝1次（未计算），改用单节点Slurm并复用原storage发布协议，未改系统或pool。lemon初始非共享cwd警告后，runner在共享冻结代码目录实际执行。bootstrap10批均数值完成，但外层重复创建summary.json导致退出1；旧报告/FAILED/原scratch保留。逐份验证原节点SHA、100个rep编号、统一列/点估计，再经10个新发布任务复制恢复，重算bootstrap0、重训0。曾有一次过早汇总和一次失败任务gate拒绝，日志均保留；包装器已修复并增加回归测试。
