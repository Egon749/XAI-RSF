# 本轮实现快照

为便于在一个文件夹内审阅模型方法，本目录收录提交37be0ee05ed64f31ba18c905795403aa8b9012ed的森林实现原文、针对性测试和参数依赖声明。source_manifest.json记录每个源路径、快照路径和SHA256。快照与仓库实际运行模块逐字节核对。

- crforest/core.py：旧有符号分数、新原因平方分数、独立退市公司支持规则、树生长与森林预测。
- crforest/partition.py：共享切点、候选变量覆盖及分区预算。
- crforest/shrinkage.py：raw直接父节点AJ收缩。
- crforest/functions.py：完整q2函数分解与保存重载。
- crforest/higher_order.py等：原有兼容实现；本轮仍只用q2。
- cause_support_checks.py：本轮14项实质检查的源文件快照，原文件是仓库tests/test_cause_support_ablation.py。
- core_changes.patch：相对于本轮来源基线51e58f的核心实现变更。
- dependencies/：实验入口的静态本地导入依赖快照，方便阅读既有评价、历史数据接口和独立核验；不包含任何企业原始数据或模型参数。

这是审阅快照，不是另一个需要维护的模型分支。实际运行仍从完整仓库根目录导入crforest和experiments；不要把本目录当作独立训练数据包。数据路径、Slurm工具、R环境及其他外部依赖见上级协议和环境文件。功能测试和历史模型重现已经完成，本次打包未新增真实森林拟合。
