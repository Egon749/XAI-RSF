# 执行和数值核验

实际拟合代码：7a8ae2b7c96791607b0a31a1e027356500a01c4e。最终完整测试：147 passed, 1 warning in 5.34s。第一次测试未配置独立R入口，142通过、5跳过；补齐既有R环境后训练前147项全部通过，0跳过，原日志均保留。既有EBM缺失值绘图提示不是本轮模型错误。

两节点Slurm均申请并检查。小试作业2571，实际主机{'lemon': 2}；正式作业2572，实际主机{'oliverdesktop': 6}。任务为单线程、4GiB进程上限，由现有cgroup／调度控制亲和，未绕过P核第二线程限制，也未为增加占用率重复训练。

| stage | task | model | seed | host | seconds | cpu_seconds | peak_rss_MiB | child_peak_rss_MiB | new_forest_fits | leaf_updates | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pilot | 0 | xai_q2 | 42 | lemon | 0.445699 | 0.444669 | 153.320312 | 0.000000 | 1 | 1 | COMPLETE |
| pilot | 1 | standard_rsf | 42 | lemon | 1.167027 | 0.080848 | 156.937500 | 245.812500 | 1 | 0 | COMPLETE |
| formal | 0 | xai_q2 | 42 | oliverdesktop | 3.427479 | 3.426587 | 218.082031 | 0.000000 | 1 | 1 | COMPLETE |
| formal | 1 | xai_q2 | 101 | oliverdesktop | 3.384316 | 3.384159 | 214.894531 | 0.000000 | 1 | 1 | COMPLETE |
| formal | 2 | xai_q2 | 202 | oliverdesktop | 2.710953 | 2.710864 | 213.351562 | 0.000000 | 1 | 1 | COMPLETE |
| formal | 3 | standard_rsf | 42 | oliverdesktop | 1.593845 | 0.056214 | 155.722656 | 245.597656 | 1 | 0 | COMPLETE |
| formal | 4 | standard_rsf | 101 | oliverdesktop | 2.496042 | 0.088136 | 155.734375 | 246.125000 | 1 | 0 | COMPLETE |
| formal | 5 | standard_rsf | 202 | oliverdesktop | 1.561888 | 0.056250 | 155.648438 | 245.726562 | 1 | 0 | COMPLETE |


正式6次拟合（q2三个、RSF三个），另3次固定叶估计更新；小试2次4树拟合及1次叶更新。正式重复训练计算记录为390×6=2340条，独立训练公司仍390家，独立先退市事件49个。正式任务全部计算、回传和哈希验证通过，失败0、重训重试0。

一次回执读取检查误以为hetero-run报告在原清单旁，出现FileNotFoundError；实际计算已成功。已增加显式--report并逐项验证任务身份后复制回执，无拟合重跑，无模型算法变化。冻结模型代码仍为上述SHA，交付代码包含此编排修正。

独立重放1948个节点、1022个叶估计，直接风险集递推AJ和父／叶凸组合最大差0.000e+00。三个RSF公司bootstrap和训练中位数均核对。保存重载、CIF非负／单调／两原因和≤1全部通过，没有裁剪预测或放宽门槛。森林没有梯度优化器，Hessian和数值gap不适用。
