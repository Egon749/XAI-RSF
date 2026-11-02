# 计算、来源和发布核验

本轮研究分析新增森林拟合0、校准器拟合0、其他模型拟合0，不作超参数或倍率搜索。分析入口显式阻止CompetingRiskForest.fit，新增分析代码没有调用fit/fit_transform/fit_predict。用户要求的完整pytest会运行既有合成测试用估计器；这些是测试夹具，不是本轮真实研究拟合，未生成新的企业预测模型。

完整测试命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CRFOREST_RSCRIPT=/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript \
EW_RLIB=/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib \
/home/oliver/.venvs/xai-baselines-20261027/bin/python -m pytest -q
```

最终结果：**251 passed、0 failed、0 skipped**，1条既有EBM缺失值可视化警告。独立R的AJ、竞争风险分裂和加权AJ核对均在完整套件中实际运行。初次和最终日志分别保留TESTS_full.log及TESTS_final.log，未删失败测试、未放宽阈值。

新增7项实质性检查：无事件年贡献0对、并列预测半计数、单一年份退化为普通AUC、全无事件保持NA、配对权重不同于等权均值、原随机流与逐年Top20名额一致、风险时间在首次事件截止且正确处理1月1日、隐私抑制区别于零值。这里“7项”按pytest函数计数；部分函数同时覆盖多个性质。

实际工件核验：

- 用户指定8个年内退市AUC四舍五入值全部精确复现。
- 160项逐年检查中，sklearn、原fast_metrics与独立纯配对计数一致；未定义情况一致保留NA。
- 36次历史正式任务的72份raw/收缩保存预测读取后，三种子平均逐元素复现原endpoint风险；没有调用森林预测生成新数据。
- 所有被支持窗口的结局已知，IPCW权重实际为1；原pooled AUC/AP/Brier全部复现。
- 1000次索引与原cause-support文件逐元素相同，所有臂/版本/原因/时限共用；896行区间完整保存，重抽样有效数加未定义数均为1000。
- 2025年24个月16行指标均NA；不进行该窗口的结局bootstrap。
- 443条唯一风险历程、207009风险天、62次先退市及251次先恢复按日历分割前后精确守恒。停止时点只用保存的事件/删失信息，不扩展随访。
- 45条已有终止依据与旧审计的事件类型、事件公告停止日逐条一致；43条明确依据、2条原文本不明确，其余17条没有已审查依据，全部保留缺口。
- 594个已读取来源文件哈希前后核对；另有终止日期核对来源哈希列在FINAL_VALIDATION.json。112个其他分支文件用git show读取并逐字节匹配固定提交。
- 除AGENTS新增授权外，2259个既有跟踪文件逐一与来源提交Git对象比较一致，包括CURRENT指针、旧冻结协议、模型规范和历史报告；当前研究参考保持。
- 公开CSV列、全部文本中的完整公司ID和公司预测文件扩展名检查通过；未新增发布逐公司数据。稀少日历/原因格采用合并和互补抑制，完整表仅私有。
- PNG及从PDF实际渲染的图都已人工检查中文、数值、NA标记和布局。早期PDF的CFF/type42字体警告已通过Type3嵌入修正；最终Poppler日志为空。旧图版本保留在本轮私有audit目录。

失败与恢复明细见FAILURES_AND_RECOVERIES.json。两次首次统计入口因worker评价文件不含groups而失败，改为从保存标签读取公司组；未改任何输入。数字验收本身没有失败。没有未报告的不利实验臂，没有新的真实训练任务，也没有Slurm作业需要等待。本轮指标统计约16秒，漂移统计小于1秒，轻量只读分析直接在oliver运行。

执行代码快照保留在各成功attempt的executed_source_snapshot，其analyze.py字节哈希与该次stage_receipt记录完全一致；补充的自动快照功能服务未来复现，不改变本次统计值。原始已执行版本、补充pooled AJ后的版本及失败attempt均保留。
