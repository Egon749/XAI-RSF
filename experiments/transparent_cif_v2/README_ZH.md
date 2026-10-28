# 透明CIF模型更新：成对候选＋全局凸权重

本轮重点是**预测器本身可以完整展开、逐项核算和独立执行**，不是给黑箱增加一个解释器。基于固定提交 `5de9cf55a40de6b8af41a6877e51ab3529c6c003`，只新增可选模块，未修改原森林、函数编译器、旧配置或历史实验。原有默认预测完全保留。

## 1. 现在部署的究竟是什么

`TransparentCIF.predict`直接执行完整原生CIF展开：

`F_k(t|x) = mu_k(t) + sum_j f_jk(t,x_j) + sum_(j,l) g_jlk(t,x_j,x_l)`。

所有函数都由实际树及同一个全局权重向量代数编译，模型部署包无需原森林，也不会调用隐藏残差、教师、门控或SHAP代理。原生输出在**概率尺度**，不是newRSF2的log-hazard。正负分量不是单独合法概率，也不是现实因果效应。参考分布为固定结构训练样本的边际乘积分布；真实联合支持另列。

`complete_tables`导出全部主效应与二元表、真实阈值、缺失单元、每格共同观察支持及所有原因的函数值；`explain`给出某条记录的全部函数贡献和基准、CIF及生存概率。不能只显示前5项就叫完整解释，也不能把函数数当成人类理解效果。

## 2. 两个可独立消融的变化

### 成对候选树库

`crforest/pair_candidates.py`在原q2森林之外增添少量深度2的候选树，每条路径只用两个变量。联合评价四个叶区域，因此不要求两个变量先各自存在强边际作用。候选来自结构训练集：一部分复用已有路径变量对，另一部分由固定随机流探索未覆盖组合；有限候选、有限共享切点、有限新增树数全部保存。每棵候选有独立公司bootstrap；四个叶都必须满足原记录数、公司组和事件支持门槛。叶估计仍然是Aalen–Johansen。

联合分区分数是 `sum_cell (n_cell/n) * logrankCR(cell vs rest)^2`，是搜索启发式，不是统计检验或因果交互发现。使用全部原因，不丢弃竞争风险。没有宣称穷尽所有变量对或阈值。

这是**额外候选银行**，不是替换原每个节点的分裂准则。原树完整保留。为了区分配对机制与单纯增加树数，还生成相同新增数量的普通树对照。全局空间格预算ledger尚不支持这种附加树模式，代码明确拒绝组合，而不是悄悄绕过约束。

### 一套全局非负权重

`crforest/transparent.py`在公司组与树结构训练完全分离的历史留出集上学习：

`F_k(t|x)=sum_b w_b F_bk(t|x), w_b>=0, sum_b w_b=1`。

**同一套权重适用于全部企业、全部时刻及全部原因**。禁止按输入、时间或原因更换权重；禁止按交互出现频次乘B/m。

权重目标为IPCW多状态Brier（目标、竞争、无首次事件三个状态），加向等权的固定L2偏离成本。企业接口只用有来源资格的12个日历月端点；提前删失和来源未明结局不标负例。内核也支持显式给定其他期限的资格合同，但不从12月标签猜6月资格。

删失分布只用结构训练资料估计；权重参数只看内部留出资料；评价期不参与。使用SLSQP解析梯度＋可选Frank-Wolfe清理，并单独检查凸目标的FW gap。未收敛直接报错，不能把未解决解或悄悄回退等权当作成功。

凸组合保持各CIF的非负、时间单调及所有原因概率和不超过1；线性组合不提高原函数阶数。编译器和独立逐树预测逐值核对。这里的性质不等于概率校准、统计可辨识、预测优势或算法首创。

## 3. 六个匹配方案

|银行|等权|学习全局权重|
|---|---|---|
|原q2银行|base_equal|base_weighted|
|增加同样数量普通树|ordinary_extra_equal|ordinary_extra_weighted|
|增加成对候选树|pair_equal|pair_weighted|

三种银行使用同一结构训练公司，权重版本使用同一内部留出公司。`ordinary_extra`与`pair`总树数相同；它们的候选搜索计算量并不相同，必须单独报告。原full-data q2及M2只作参考，不冒充相同训练角色下的对照。

保留原种子42/101/202，但不挑最好种子。所有输出只写新目录。本次不恢复gamma、不增大q、不修改事件标签和样本入口。

## 4. 重要限制：凸权重并非万能修复

若不同树分别只估计某个完整主效应和另一个完整交互，凸组合通常不能把两者都原幅度相加。新增配对候选也不保证真实交互能被恢复。`demo.py`因此同时保留等权、真实标签内部留出权重，以及**仅在模拟中用真CIF求解的乐观容量诊断**。真值权重不是可部署方法，不能与普通训练结果混作性能胜利。

合成结果必须完整保留，包括未恢复幅度。真实企业是否改进，等待服务器重新运行；本更新没有企业性能结果。

## 5. 输入与运行

读取先前 `interpretable_baselines_v1` 私有运行的 `input/`，其中必须有：

- final_train_PRIVATE.npz / final_train_metadata_PRIVATE.parquet；
- final_eval_PRIVATE.npz / final_eval_metadata_PRIVATE.parquet；
- plan.json，含forest_configs.q2；
- legacy/原统一评价代码。

另外提供已有75列的`features.json`：每列name、unit、definition、transform。顺序必须与旧输入一致；未知单位标明UNVERIFIED，不能编造。不要为本轮重新搜行业表或重建整个风险集。

默认Parquet保持原数据形式。仅为明示的等值导出和低依赖测试，允许`--metadata-format csv`；不会自动用CSV替代缺失Parquet，也不能更换为未经核验的来源。

```bash
python -m pip install -e '.[dev,benchmarks]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# 在服务器设置既有CRFOREST_RSCRIPT和EW_RLIB后运行全部测试，不跳过R检查。
python -m pytest -q
python -m experiments.transparent_cif_v2.demo --output /全新私有目录/demo

python -m experiments.transparent_cif_v2.run train \
  --source-input /已核验interpretable_baselines运行/input \
  --features /已有私有字段合同/features.json \
  --seed 42 --tree-workers 1 --ebm-control \
  --output /全新私有目录/seed42_fit

python -m experiments.transparent_cif_v2.run predict \
  --source-input /同一运行/input \
  --trained /全新私有目录/seed42_fit \
  --output /全新私有目录/seed42_predictions
```

101、202两个种子按同样冻结设置执行。用现有Slurm并行独立种子；单进程不会自动跨机。`tree-workers`不得超过实际分配的物理核。成对候选库本版串行搜索；不能据墙钟宣称全硬件利用。

```bash
python -m experiments.transparent_cif_v2.evaluate \
  --source-input /同一运行/input \
  --predictions /全新私有目录/seed42_predictions /全新私有目录/seed101_predictions /全新私有目录/seed202_predictions \
  --output /全新私有目录/evaluation
```

评价重用同一个legacy endpoint guard、Top排序、事件去重和提前量函数；不修改名单打破并列方式。另行复用既有共同公司bootstrap程序获得条件区间。evaluate本身只产生点估计，不假称已经做bootstrap。

## 6. EBM对照修复

`--ebm-control`启用端点EBM主效应和10对交互两种配置，仅在同一公司组隔离的**历史内部留出集**选择50/200/500轮，以端点IPCW Brier选择；保留全部候选诊断。不用2023/2024选择轮数，不再对2000轮固定无验证结果作强基线优越性声明。

这是公司隔离的内部验证，**不是时间前推验证**，也不是充分调优的最优EBM。后续正式时间验证需要另行冻结协议。此基线仍是12月分类器，其解释在log-odds尺度，没有完整竞争风险曲线。保留已选模型，不将留出集混回去重拟合。未安装InterpretML时该显式选项会报依赖错误，不悄悄跳过。

## 7. 本轮Codex应报告什么

重点是完整透明预测器，而非又加一套事后解释：

- 所有主函数、交互、阈值/分区、缺失与零支持格，完整解释的误差；
- 概率合法性、全时间/全原因共享权重、重载一致性；
- 有效树数和权重集中度，候选对覆盖率及联合分区支持；
- 模拟真实主效应/交互的幅度恢复，而不仅是结构F1；
- 同树数、同角色的普通树与配对树、等权与学习权重的公平对照；
- AP/AUC/Brier以及Top20/25的企业级已确认命中占比、核查公司数、独立事件数、3/6月提前量和未知负担；
- 函数规模和解释稳定性，不能仅用加和正确宣称可靠；
- 不得称已经查看的2023/2024为独立盲测，数据继续PROVISIONAL。

不需要所有预测指标超过黑箱才有价值，但不能用解释透明掩盖命中率低；也不能用提高AP掩盖完整解释规模失控。
