# 运行与复核入口

本轮私有运行目录：`/home/oliver/MainDiskData/xai_rsf_temporal_2023_2024_v1_20261027_sWV4S5D1`。冻结训练提交：`5bf5be43ed12fa027dd42c9222b852ee52a21ec9`。后续评价/报告代码不改变训练快照。

`prepare.py --run NEW_UNIQUE_DIRECTORY`恢复已冻结最终样本与真实来源时钟，校验历史V20预处理；应使用新的唯一目录，不能覆盖已有运行。依赖的原私有数据路径在脚本常量中明确列出。

`stages.py --run DIRECTORY --repo REPOSITORY --stage freeze`要求干净Git状态，将提交归档为只读任务输入，生成两个试运行及48个独立四树批次清单。`launch.py`选择各节点已验证的Python环境；`worker.py`仅允许final折并调用原训练引擎。树的全局编号与种子不因节点分配改变。

按序使用现有`hetero-submit DIRECTORY/pilot.jsonl physical both`，完成后执行`stages.py --run DIRECTORY --stage pilot --job JOB_ID`。只有数值/重载/发布核验通过，才提交`training.jsonl`。训练完成后运行training gate，生成六项assembly清单；完成assembly gate后再运行`evaluate.py --run DIRECTORY`。每个任务有独立输出所有权；发布失败应依照pool receipt恢复发布，不能自动重新拟合。

每个清单同时有`*.py.hetero.json`侧文件，支持既有hetero-run入口。实际本轮显式使用physical/both，避免把P核SMT重复计入。每项单线程，两台机器通过独立树批次并行；单个Python进程不跨机器。

评价阶段使用仓库为PYTHONPATH，并将OPENBLAS_NUM_THREADS、OMP_NUM_THREADS、MKL_NUM_THREADS设为1，PYTHONDONTWRITEBYTECODE设为1。代码先验证模型/预测哈希及样本索引，再生成汇总CSV与执行回执。模型、预测、公司解释、原始切点和完整函数系数留在私有运行目录。

本轮固定完成一次，不自动追加参数搜索、重新筛种子或按2023/2024表现修改模型。聚合代码的修改应保留旧评价版本及结果，不能覆盖完成的execution_receipt。
