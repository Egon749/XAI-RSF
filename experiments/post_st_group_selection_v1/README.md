# 有限分组筛选实验

先读frozen_protocol.json、DATA_AND_PROTOCOL_AUDIT_ZH.md和RESULTS_ZH.md。执行入口：`python -m experiments.post_st_group_selection_v1.run prepare|freeze|pilot|batch|diagnostics`，汇总入口：`python -m experiments.post_st_group_selection_v1.deliver`。

私有路径见run_location.json，源数据沿用原实验，不复制公司文件到Git。复现必须先在run.py指定新的唯一RUN，prepare拒绝已有冻结目录；完成测试并提交后freeze建立不可变代码快照。pilot两项纳入总150项，batch只运行其余148项，避免重复。每阶段JSONL及hetero sidecar由运行器生成，既有两节点Slurm/NFS调度并核验回传。

支持门槛关闭时监督候选搜索不执行，不能把回退拟合写成筛选成功。当前实现只实际验证此已冻结数据下的门槛/回退分支；若新的数据触发支持通过，运行器明确停止，须补齐并验证支持分支后另行冻结，禁止静默使用未经核验的选择逻辑。历史23候选和1408函数诊断独立保存，不影响正式回退决定。

中文报告生成入口：`python -m experiments.post_st_group_selection_v1.report`。汇总恢复不需要重训，原失败日志不得删除。
