# 日历人时线性水平更新（探索性）

结论与全部预设臂见 [RESULTS_ZH.md](RESULTS_ZH.md)。只处理cause-support保存的baseline收缩参考版本，不拟合森林或校准器，不替换CURRENT指针。参数来自各Jan1自身快照，评价结局仅用于固定参数后的评价。

协议在读取本轮评价标签前冻结于 `bea1601877f7efe50227ea817946cede93997ad2`；来源为 reporting-drift `bb02b951502381df53581e15173485f0a096f345`。窗口与方案是在查看已有公开日历汇总后设计的，全部结果为探索性。

## 复现

在本分支已提交且工作区干净时，使用现有Python环境：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/oliver/.venvs/xai-baselines-20261027/bin/python \
  -m experiments.post_st_calendar_level_v1.run
```

入口自动创建唯一私有目录；也可通过 `--run-dir /绝对路径/尚不存在的新目录` 指定。程序会依次复现五个参考O/E、计算自己的历史人时、检验证书、经现有两节点Slurm池完成三个独立年度的全函数检查、复用原1000组评价重抽样索引、验证哈希并生成 `deliverables/`。不会覆盖本目录已发布结果。任何失败即停止；不要重跑同一目录覆盖证据。

运行需要 `run_location.json` 指向的原cause-support私有数据、模型、原抽样索引，以及本机hetero-run和两节点既有Python环境。私有目录保存训练重抽样索引、合并森林、包络顶点、完整函数、逐公司所有时间预测和日志。公开仅保留汇总、代码与哈希，**没有扩展已有rankings发布授权**。

文件分工：`rates.py`为人时及证书规则；`prepare.py`核验快照与参考值；`verify.py`复用merge和transformed_expansion、按verify_export相同口径检查完整函数；`analyze.py`固定预测评价；`collect.py`核对输入/输出及Slurm发布；`report.py`汇总和绘图。训练接口在研究进程中被禁止调用。

## 评价与文件

- `LEVEL_SLOPES.csv`：hist/recent人时、事件支持、请求倍率和训练公司bootstrap区间。
- `CERTIFICATE_CHECKS.csv`：请求/认证斜率、证书损失；uncertified只作诊断。
- `ARM_METRICS.csv`：两原因、年内/pooled、12/24个月；cause0为两原因联合Brier之和。
- `PAIRED_DIFFERENCES_95.csv`：臂减reference；1000次固定模型/斜率、按年分层公司bootstrap；未定义次数完整保留。
- `LEVEL_SCENARIOS.csv`：低/中/高均值与观察率，非置信区间；2025年24月只有预测。
- `FUNCTION_VALIDATION.csv`：18套全函数表示、全样本/原生时间合法性、误差及诊断违规数。
- `RISK_LEVEL_COMPARISON.png/.pdf`：无公司标识的水平图。
- `PAPER_NOTE_ZH.md`、`FORWARD_VALIDATION_PLAN_ZH.md`：论文表述和冻结的未来评价规则。

`scope`是评价入组年；`cause=1`为先最终退市、`cause=2`为先恢复。支持窗口沿用原评价器且实际IPCW权重均为1。2025年24个月性能NA。年度小计1—4抑制；年度整队列风险率保留。NaN不是0；恢复2023年12月O/E为0、abs(log(O/E))为无穷，对应差值NA。

完整测试（单元测试中的小型合成拟合不计研究森林拟合）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CRFOREST_RSCRIPT=/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript \
EW_RLIB=/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib \
/home/oliver/.venvs/xai-baselines-20261027/bin/python -m pytest -q
```

当前结果不用于选择新窗口、不自动采用任何映射、不修改旧报告。已有temporal-affine独立校准只在论文说明中引用，没有重新计算。
