# 制造业实验执行入口

私有运行：`/home/oliver/MainDiskData/xai_rsf_manufacturing_2023_2024_v1_20261027_XvjGsHee`。

冻结训练提交：`206e414d05e8cd15e13429637bc6efc7f0abfcd4`。运行`prepare.py --run NEW_UNIQUE_DIRECTORY`使用已有系统Python及python_calamine，核对原始行业表并在父实验输入上筛子集。目标目录必须全新，原样本与父模型只读。

使用原`experiments/temporal_2023_2024_v1/stages.py --run DIRECTORY --repo REPO --stage freeze`归档干净提交及输入。随后依次通过既有`hetero-submit`以physical/both提交pilot.jsonl、training.jsonl、assembly.jsonl，每阶段完成后用同一stages.py的对应gate验证。training gate生成assembly清单。每个清单有hetero-run侧文件可用。片段全局tree_id和seed固定，不能以重复完整训练脚本冒充并行。

训练仍调用原launch.py和worker.py，仅数据子集不同。两台机器使用上一轮核对过的独立环境；新Python依赖没有安装到训练环境。任务输出经pool计算/发布双状态及哈希验证。

全部assembly通过后，以仓库为PYTHONPATH、各数值库单线程，运行本目录`evaluate.py --run DIRECTORY`，然后`report.py --run DIRECTORY`生成汇总材料。固定模型预测平均与种子指标平均分别输出；allsector标签绝不能去掉，否则会误称那些参考模型也在制造业重训。

不覆盖已有execution_receipt或报告交付文件。发布失败先按pool receipt恢复发布，不能自动重训。完成后保留所有模型、种子和负面结果，本轮不自动追加行业/特征/预算搜索。
