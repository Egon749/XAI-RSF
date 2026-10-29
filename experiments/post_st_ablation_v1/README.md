# 当前可解释竞争风险森林的有限机制消融

设计来源261eb15，保留其后2025实验和当前研究参考模型。2023/2024已经暴露，本轮是探索性机制分析。

先运行 `prepare --stage prepare` 核验24份历史模型、6份标准RSF外部参照和旧C输入。通过完整测试后提交代码，再执行 `prepare --stage freeze` 保存不可变代码、数据哈希、协议和18任务清单。使用 `hetero-run RUN/formal.py` 执行18次新森林拟合，每次保存raw和强度10父节点收缩版本。

从调度器返回的实际报告路径运行 `stages --stage formal --job JOB --report REPORT`，核对任务身份及结果回传。运行 `stages --stage verification_manifest` 后，用 `hetero-run RUN/verify.py` 独立验证全部60个单种子模型及20个96树平均预测器；验证不重新拟合森林。

入口均使用 `python -m experiments.post_st_ablation_v1.模块 --run RUN`。模型任务使用DATA_DIR/OUTPUT_DIR，启动器按实际节点选择已经验证的Python环境。输入和结果分开、默认NFS；进程单线程，调度器管理P/E核亲和性，不自行硬编码CPU。保留失败记录，发布失败不重新拟合。

统一评价通过 `analyze`，条件配对不确定性通过 `uncertainty`，结果交付通过 `report`。公开目录仅有汇总；原始公司、预测、切点及完整函数留在私有运行目录。冻结协议规定所有主次比较，禁止根据结果选择性省略臂或更换研究参考模型。
