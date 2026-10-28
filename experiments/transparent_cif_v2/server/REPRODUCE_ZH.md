# 本轮实际运行入口与核验顺序

本轮已完成，不需要再次训练。私有运行根路径和训练提交见results_20261028/execution_receipt.json。以下为复核/新授权复现实验的顺序；所有入口使用不存在的新输出目录，不覆盖历史。

1. 用权威提交建立隔离工作区，完整读取三个原说明；准备隔离Python、R survival/randomForestSRC、Parquet和真实InterpretML。设置CRFOREST_RSCRIPT、EW_RLIB及OPENBLAS/OMP/MKL_NUM_THREADS=1，运行`python -m pytest -ra -q`，要求无跳过。本轮98项旧＋新测试通过后，额外加入真实InterpretML端到端测试，共99项。
2. 在新私有run目录建立audit/input/results/reports，执行`python -m experiments.transparent_cif_v2.server.prepare --run RUN --parent FROZEN_BASELINE_ROOT`。它只核验/拷贝冻结输入与既有来源字段定义，不重建样本。
3. 完成代码提交后执行`prepare --run RUN --repo WORKTREE --freeze`，获得不可变代码归档、输入哈希和formal.jsonl。提交`hetero-submit RUN/formal.jsonl physical both`；本次作业2546，4项（3种子＋1模拟），不是重复执行一个任意Python来冒充并行。
4. 结束后逐项核验调度returncode、compute_completed、publication_completed及SHA256。本次4/4通过。`python -m experiments.transparent_cif_v2.server.postprocess --run RUN --job 2546`执行原evaluate、增加公司/季度核查统计、准备固定预测bootstrap和独立无森林部署。
5. `python -m experiments.transparent_cif_v2.server.validate_bootstrap --run RUN`核验2400个点估计并做非平凡重复公司对照。通过后提交`hetero-submit RUN/bootstrap.jsonl p both`，本次2547。p模式使用oliver的P物理核与lemon物理核，不启用oliver P核第二线程。1000次固定公司抽样分10批，与训练无关。
6. `python -m experiments.transparent_cif_v2.server.audit_models --run RUN --job 2546`补全所有模型的公司与原因别独立事件单元支持、完整日历案例、复杂度、数值和三种子描述性稳定性。核验完1000次唯一replicate id及发布后运行`collect_bootstrap --run RUN --job 2547`。
7. `python -m experiments.transparent_cif_v2.server.report --run RUN`只生成公开汇总；企业、真实预测、参数、阈值及个体解释留私有目录。若修订汇总错误，保留原稿并记录更正；不能因此修改模型或重训。
8. 完整测试再次通过后提交研究分支。本轮权威训练代码冻结在6a4a30ef16f3ffeb9f0d5636a065dde5996ce8d2，后续服务器报告代码不改变已冻结的模型。

所有后处理入口都针对本次固定契约，路径中的日期与固定输出目录需要在新的授权复现时显式改为新路径；不要原样覆盖已有交付。池的发布失败只能按原receipt恢复发布，不得以重训代替恢复。
