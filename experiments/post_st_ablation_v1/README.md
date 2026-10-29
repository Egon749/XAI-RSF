# 当前可解释竞争风险森林的有限机制消融

设计来源261eb15，保留其后2025实验和当前研究参考模型。2023/2024已经暴露，本轮是探索性机制分析。

先运行 `prepare --stage prepare` 核验24份历史模型、6份标准RSF外部参照和旧C输入。通过完整测试后提交代码，再执行 `prepare --stage freeze` 保存不可变代码、数据哈希、协议和18任务清单。使用 `hetero-run RUN/formal.py` 执行18次新森林拟合，每次保存raw和强度10父节点收缩版本。

从调度器返回的实际报告路径运行 `stages --stage formal --job JOB --report REPORT`，核对任务身份及结果回传。运行 `stages --stage verification_manifest` 后，用 `hetero-run RUN/verify.py` 独立验证全部60个单种子模型及20个96树平均预测器；验证不重新拟合森林。

入口均使用 `python -m experiments.post_st_ablation_v1.模块 --run RUN`。模型任务使用DATA_DIR/OUTPUT_DIR，启动器按实际节点选择已经验证的Python环境。输入和结果分开、默认NFS；进程单线程，调度器管理P/E核亲和性，不自行硬编码CPU。保留失败记录，发布失败不重新拟合。

统一评价通过 `analyze`，条件配对不确定性通过 `uncertainty`，结果交付通过 `report`。公开目录仅有汇总；原始公司、预测、切点及完整函数留在私有运行目录。冻结协议规定所有主次比较，禁止根据结果选择性省略臂或更换研究参考模型。

## 可复现执行顺序

在已具备原私有工件的服务器中使用独立、空的RUN目录。先用仓库完整测试（含已有独立R环境）核验，再提交代码以满足freeze的干净快照要求。以下JOB/REPORT分别取自两次hetero-run的实际返回值；返回作业号不代表成功，必须完成gate。

```bash
python -m experiments.post_st_ablation_v1.prepare --stage prepare --run "$RUN"
python -m experiments.post_st_ablation_v1.prepare --stage freeze --run "$RUN"
hetero-run "$RUN/formal.py"
python -m experiments.post_st_ablation_v1.stages --stage formal --run "$RUN" --job "$FIT_JOB" --report "$FIT_REPORT"
python -m experiments.post_st_ablation_v1.stages --stage verification_manifest --run "$RUN"
hetero-run "$RUN/verify.py"
python -m experiments.post_st_ablation_v1.stages --stage verify --run "$RUN" --job "$VERIFY_JOB" --report "$VERIFY_REPORT"
python -m experiments.post_st_ablation_v1.analyze --stage metrics --run "$RUN"
python -m experiments.post_st_ablation_v1.analyze --stage functions --run "$RUN"
python -m experiments.post_st_ablation_v1.uncertainty --run "$RUN"
python -m experiments.post_st_ablation_v1.final_audit --run "$RUN"
```

上述各阶段采用排他创建，不覆盖旧结果。拟合完成而回传失败时按heteropool发布恢复协议处理，不重新拟合。保存两节点环境和完整测试日志后使用report生成汇总交付；本次report包含实际作业与一次后处理修复的叙述，重现新运行时须按新receipt更新运行事实，不复制本次执行次数/作业号充当新运行证据。

本次实际运行已结束，完整收据见execution_receipt.json。主结果在RESULTS_ZH.md，论文段落在PAPER_METHODS_RESULTS_ZH.md。当前参考保留；没有执行2025检验或后续搜索。
