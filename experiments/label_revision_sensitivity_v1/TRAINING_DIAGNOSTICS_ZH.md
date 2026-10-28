# 实际执行与数值诊断

新增正式企业拟合 **6/6**，复用L0 **6/6**，新增L0补训0，试运行0，正式训练/发布失败0，模型恢复训练0。48个分片共192棵树完整合并为六个森林；无丢弃seed。

| stage | tasks | seconds | hosts | failed_tasks |
| --- | --- | --- | --- | --- |
| control | 6 | 1.4278 | {'oliverdesktop': 6} | 0 |
| train | 48 | 116.991 | {'lemon': 13, 'oliverdesktop': 35} | 0 |
| assemble | 6 | 12.0449 | {'lemon': 2, 'oliverdesktop': 4} | 0 |
| explain | 8 | 33.0203 | {'oliverdesktop': 8} | 0 |
| bootstrap | 10 | 30.6702 | {'oliverdesktop': 10} | 0 |


两节点physical池：oliver可用8P+16E、lemon8物理核，单任务1CPU、BLAS1；不声称单进程跨机器。训练实分配oliver35分片、lemon13分片。六模型合计fit CPU 2657.545秒；训练阶段墙钟116.991秒，不含准备/排队/解释/评价。合并期最大RSS 248.594MiB。训练分片沿用原6GiB地址空间上限；没有单独采集实际fit峰值RSS，Slurm accounting被禁用，不能把合并RSS冒充训练RSS。

| variant | seed | fit_cpu_seconds | peak_rss_MiB | leaf_zero_target_company_support | leaf_zero_competing_company_support |
| --- | --- | --- | --- | --- | --- |
| coverage | 42 | 94.407 | 246.914 | 35 | 40 |
| unrestricted | 42 | 786.957 | 222.578 | 25 | 341 |
| coverage | 101 | 133.877 | 248.594 | 43 | 46 |
| unrestricted | 101 | 846.693 | 224.668 | 16 | 345 |
| coverage | 202 | 96.9198 | 248.492 | 36 | 54 |
| unrestricted | 202 | 698.691 | 232.426 | 21 | 385 |


最少3事件是任意原因合计，并不保证每叶都有每个原因支持。上表零原因支持叶如实保留。无优化器/迭代损失，本模型是有限候选生存分裂与非参数叶CIF，因此不虚构数值gap或收敛状态；所有概率/保存重载/分解门槛保持原值。

测试：原训练版本完整45项（含R survival独立AJ、randomForestSRC单树分裂/叶AJ）；相关旧审计/日历/未知结局64项；本轮数据/真实抽样/交接12项，最终全部通过、无skip。初始45项又在明确冻结源码cwd下完整重复一次以消除导入路径歧义，记录未删除。初始补丁7项包含于最终12项。两次原测试合计Python fit调用86次：54次合成拟合成功、32次预期非法输入拒绝；另2次合成R单树拟合。均不计入六个正式企业拟合。

保留两项非训练中止：旧评价测试首次缺POLICY_RUN环境路径而在collection失败，补齐后64通过；发布核验脚本首次按错误层级读取publication_completed，修正为实际收据顶层后通过。无模型重训，原失败日志保留。初版交接plan包含旧协议来源字段，冻结前精简并刷新所有source_hashes，旧草稿保留，不用于任何训练。
