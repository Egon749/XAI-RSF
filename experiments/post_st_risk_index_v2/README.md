# 风险指数 v2：真实验证与当年公司袋外阈值

本目录完成真实数据验证，不拟合真实森林或校准器。源分支/提交为 `research/xai-rsf-risk-index-v1` / `fcf98aac4c68ac4ac1cadac4227262e0937e94a3`；仅在 `research/xai-rsf-risk-index-v2` 发布，不合并、不替换参考模型。

先读 [RESULTS_ZH.md](RESULTS_ZH.md)，使用解释见 [USER_GUIDE_ZH.md](USER_GUIDE_ZH.md)，论文草稿见 [PAPER_NOTE_ZH.md](PAPER_NOTE_ZH.md)。v1 因 2019 历史退市分母为零而停止的记录完整保留。

## 冻结与执行

- 用户 2026-11-02 明确修订阈值来源：当年参考森林的公司袋外预测，取代跨历史森林的样本外预测。
- 协议提交 `7abd471f884b0c26592f9afe77650edc32e4a02c` 早于本轮评价标签访问；真实 A/B 执行代码为 `97fef1bdfe4649791f7805a1fe9b26f1534fddd6`。
- 只读取 cause-support 的 baseline/shrunk 参考：83 列、q2、深度 4、直接父 AJ 收缩 10、42/101/202 各 32 棵树。
- A：3 个年度的指数、排序、完整分解、重载及逆变换；全局 A 门通过后才执行 B。
- B：3 个年度逐种子/原树号重建 bootstrap，核对每个节点及父节点收缩，再计算 OOB 阈值。B 输入快照中移除了复制的评价文件，原始数据未动。
- 6/6 个真实验证任务完成；Slurm 2622、2623。两节点池提交成功，实际六个短任务均由 oliver 执行；没有把两个节点的分配写成两个节点都完成计算。
- `forest_fits=0`、`calibrator_fits=0`；测试夹具中的合成 refit 单列，不属于企业研究拟合。

## 代码与复用

`prepare.py` 校验原工件/时点/哈希及五个 O/E 门，冻结副本和原 bootstrap 分析索引；`run.py` 负责 A→全局门→B 依赖；`worker.py` 完成真实数值验证；`oob.py` 负责抽样重放与纯预测阈值函数；`analyze.py` 固定阈值评价、原 1000 抽样及互补小格抑制；`publish.py` 仅发布已抑制汇总和图。

复用 v1 的 `index.py` 和 `tests/test_risk_index.py`：正分母、线性指数、逆变换、纯分位数、分档以及尺度不变性。函数 `historical_thresholds` 的旧名称保留兼容，v2 实际传入的只有当年 OOB 指数，不读取历史森林、历史 OOF 或训练内预测。新增 `tests/test_risk_index_v2.py` 检验 OOB/重复公司、合成结局变动 refit、阈值不接收结局和小格互补抑制。未改 `crforest/`。

## 复现命令（服务器；只读既有模型）

从已提交、干净的本分支工作区执行；环境和私有源见 `run_location.json`、`execution_receipt.json`。再次运行必须创建新目录，不能使用本次已完成目录。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CRFOREST_RSCRIPT=/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript
export EW_RLIB=/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib
RISK_INDEX_PY=/home/oliver/.venvs/xai-baselines-20261027/bin/python
"$RISK_INDEX_PY" -m pytest -q
export RISK_INDEX_RUN=$(mktemp -d /home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj/risk_index_v2_repeat_XXXXXXXX)
"$RISK_INDEX_PY" -m experiments.post_st_risk_index_v2.prepare
"$RISK_INDEX_PY" -m experiments.post_st_risk_index_v2.run --prepared-run "$RISK_INDEX_RUN"
"$RISK_INDEX_PY" -m experiments.post_st_risk_index_v2.analyze --output-name analysis
"$RISK_INDEX_PY" -m experiments.post_st_risk_index_v2.publish --analysis-name analysis --public-output "$RISK_INDEX_RUN/public_summary"
```

`hetero-run` 必须可用；两个节点使用各自验证过的解释器。实际本轮在 `analysis_final/` 保存最终汇总；第一次 `analysis/` 留存，仅修正了未支持终点的状态元数据（不把未执行的 bootstrap 写成 1000 次失败，不把未评价的指标一致性写为 False），不改变分数、阈值、分档或任何可估计结果。

`verify.py --output NEW_FILE.json` 对本次公开目录执行原输入不变、原报告不变、数值门、R 测试执行、小格抑制和公司标识扫描。需要先用完整测试的 `--junitxml` 将 XML 留私有运行目录 `pytest_final_PRIVATE.xml`；不发布该 XML。报告代码首次将 `independent_risk_sets` 误匹配为 R 检验，已改为 5 个准确测试名并留存这次后验审计失败记录，研究计算不受影响。

本次发布前最后一轮完整测试为 `TESTS_release.log`（268 passed），对应私有 `pytest_release_PRIVATE.xml`；核验时用 `verify.py --tests-xml PATH --output NEW_FILE.json` 指定该 XML。此前日志也保留。小格测试使用完全虚构的 1900 年单元格，不使用任何真实三档小格值。

## 输出索引与隐私

|文件|内容|
|---|---|
|`INDEX_SUMMARY.csv`|训练/评价两集合、两原因/时限的指数分布|
|`RANK_IDENTITY_CHECKS.csv`、`WITHIN_YEAR_IDENTITY.csv`|逐年和年内加权指标完全一致；无事件 AUC 与随访不足保留 NA|
|`EXACTNESS_CHECKS.csv`|所有样本/原生时间的全分解、独立逐项求和、重载、逆变换、基线偏差|
|`OOB_SUPPORT.csv`|树数、有效公司数、全部节点及标签替换核验；仅阈值工件哈希|
|`TIER_PERFORMANCE.csv`、`TIER_DIFFERENCE_95.csv`|三档两原因发生率/提升及固定模型阈值条件下的高减低区间|
|`TIER_SHARE.csv`、`TIER_MONOTONICITY.csv`|人数比例、相对 50/30/20 的偏离、发生率单调性|
|`HIGH_REST_PERFORMANCE.csv`、`TOP20_OVERLAP.csv`|预先声明的高/其余合并补充表及原 Top20 重合|
|`GROUP_POINTS_SUMMARY.csv`|变量组的绝对得分之和的均值，另列组内带符号求和再取绝对值的均值|
|`REFERENCE_GATE.csv`|原退市 O/E 复现|
|`TIER_INCIDENCE.png/.pdf`|可公开发生率图；S 为抑制，NA 为未支持，均不当作 0|

本次私有目录由 `run_location.json` 指向。`results_A/` 中含完整分解、μ、逐公司 CIF/指数和重载工件；`results_B/` 中含 OOB、阈值、bootstrap 重复次数；`analysis_final/` 中含完整三档统计/差值/图。公司标识、真实切点、μ、阈值和逐公司数据没有复制到 Git。旧 rankings 授权未扩展。

三档中出现 1–4 个事件时，互补抑制该三档全部事件数/率/提升/相关差值；人数出现 1–4 时同样抑制人数及比例。高与其余的预设二分报告单独检查两格支持。CSV 中 `publication_status` 区分隐私抑制与结局未支持。计数为 0 可报告，空档发生率不能填 0。

2023–2025 均为已暴露队列；PIT 原始披露版本沿用既有 PROVISIONAL 限制。指数未改善概率校准，分档不是新模型或独立时间外验证。
