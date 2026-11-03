# 模型代码与复现

`crforest/` 是来源提交 `d279f889a23bda3e779b434dd230d484b0269420` 的逐字节代码快照，并保留其全部模块依赖；本模型只使用配置中的q2、shared_quantile、balanced、signed_sum及支持阈值2。`核验/CODE_SNAPSHOT.json` 记录代码哈希。

- `core.py`：公司bootstrap、竞争风险分裂、退市支持检查、森林预测与保存加载。
- `partition.py`：路径变量预算、均衡25变量候选和共享分位数切点。
- `shrinkage.py`：直接原始父节点AJ收缩。
- `functions.py`：完整基线、主效应、二元交互函数编译、预测和逐项贡献。
- `main.py`：使用上述原实现的独立固定配置入口，不实现另一套森林。
- `原实验入口/`：保留实际作业使用的worker、ensemble、launch、common源文件用于追溯；它们的包导入路径仍对应仓库根目录的experiments，不能直接在此目录运行。
- `plot_saved_rankings.py`：生成已展示图片的实际绘图脚本，依赖完整仓库中的评价工具；只有读取预测、画图和写出私有文件，没有训练调用。

## 环境

实际环境版本在上一级`配置/environment.json`。可在独立Python环境安装此目录的包：

```bash
python -m pip install -e '主模型/代码[dev]'
```

绘图额外需要该环境记录的matplotlib；中文字体使用Noto Sans CJK。

## 输入与输出

输入NPZ中`X`必须是74列，严格遵循`配置/INPUT_FEATURES.csv`的input_position顺序；允许NaN，不重新填补。训练文件还需对齐的`time`（天）、`event`（0删失/1先最终退市/2先恢复）、`groups`（公司组）。评价输入只需`X`及`sample_ids`，训练不读取评价结局。数据准备、披露时钟、随访截断必须沿用冻结的私有输入，入口不会自动恢复缺失原始数据。

输出目录必须全新；包含公司信息、模型参数与真实切点的输出只能放私有服务器目录。此代码交付不包含企业数据或拟合参数。

## 保存模型的预测和完整核验

```bash
python 主模型/代码/main.py predict \
  --model /私有路径/96树收缩森林.json \
  --input /私有路径/eval_PRIVATE.npz \
  --days 365 366 730 731 \
  --output /私有路径/新的预测目录

python 主模型/代码/main.py verify \
  --model /私有路径/96树收缩森林.json \
  --train /私有路径/train_PRIVATE.npz \
  --eval /私有路径/eval_PRIVATE.npz \
  --saved-predictions /私有路径/predictions_PRIVATE.npz \
  --output /私有路径/新的核验目录
```

verify在全部训练与评价X、全部原生时间及四个评价日偏移上，检查森林与完整函数预测、逐项贡献求和、保存重载和CIF合法性；容差1e-10，无clipping。它不拟合森林。实际日历终点由每家公司的起点确定，不能将365天预测一律当作12个日历月。

## 需要重现训练时的固定入口

下列命令会训练新模型，本次整理交付没有执行。仅在得到新的训练授权时，在新私有目录运行；正式独立年度/种子任务应使用两节点Slurm池。

```bash
python 主模型/代码/main.py train \
  --train /私有路径/train_PRIVATE.npz \
  --eval /私有路径/eval_PRIVATE.npz --seed 42 \
  --output /私有路径/新的种子42目录
```

分别使用42、101、202；每次固定32树、单线程，先保存raw再生成AJ10收缩版本，两个版本都验证完整解释。不要对收缩模型再次收缩。

```bash
python 主模型/代码/main.py merge \
  --models /私有路径/种子42/shrunk/forest_PRIVATE.json \
           /私有路径/种子101/shrunk/forest_PRIVATE.json \
           /私有路径/种子202/shrunk/forest_PRIVATE.json \
  --train /私有路径/train_PRIVATE.npz --eval /私有路径/eval_PRIVATE.npz \
  --output /私有路径/新的96树目录
```

merge不训练，使用同一年同一训练快照的三个收缩森林，核验96树预测等于三种子预测均值并导出完整函数。不同年度不能混合。

## 复现风险图

在完整仓库根目录、已安装依赖的环境中：

```bash
PYTHONPATH=. python 主模型/代码/plot_saved_rankings.py \
  --run /私有路径/drop9_support运行目录 \
  --output /私有路径/新的绘图目录
```

绘图读取原始CIF，不归一化、不放大、不使用当批样本调整分数。支持的年度/时限为2023和2024的12/24个月及2025的12个月。
