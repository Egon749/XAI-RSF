"""Render public aggregate reports only; never publish anchor arrays or raw models."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'normalized_rmse':'归一化贡献 RMSE','rmse':'原始贡献 RMSE','importance_spearman':'重要性排序 Spearman','top10_jaccard':'Top10 变量 Jaccard','sign_agreement':'贡献符号一致率','median_anchor_correlation':'企业内贡献相关性中位数'}
ENGLISH={'normalized_rmse':'Normalized attribution RMSE','rmse':'Raw attribution RMSE (probability)','importance_spearman':'Importance rank Spearman','top10_jaccard':'Top-10 variable Jaccard','sign_agreement':'Attribution sign agreement','median_anchor_correlation':'Median within-anchor correlation'}
def table(df,anchored=False):
    header='| 指标 | q2 均值 | 黑箱均值 | q2−黑箱 | 95% 条件估计区间 | 有利方向 |\n|---|---:|---:|---:|---|---|\n'
    for _,x in df.iterrows():
        header+=f"| {LABELS[x.metric]} | {x.q2_mean:.6f} | {x.blackbox_mean:.6f} | {x.estimate:+.6f} | [{x.ci_low:+.6f}, {x.ci_high:+.6f}] | {'负' if x.metric in ['rmse','normalized_rmse'] else '正'} |\n"
    return header

def main(run):
    out=run/'reports';s=pd.read_csv(out/'stability_intervals.csv');a=pd.read_csv(out/'original_anchor_intervals.csv');bg=pd.read_csv(out/'background_intervals.csv');d=pd.read_csv(out/'task_diagnostics.csv');native=pd.read_csv(out/'native_q2_stability.csv');r=json.loads((out/'execution_receipt.json').read_text());p=s[s.n_replicates==50];primary=p[p.metric=='normalized_rmse'].iloc[0]
    direction='支持 q2 的归因更稳定' if primary.ci_high<0 else '支持黑箱的归因更稳定' if primary.ci_low>0 else '尚不足以明确区分两者的归因稳定性'
    report=f'''# 50 次配对公司组重训：稳定性差值

**主要结果：在本次固定条件下，{direction}。** 这是固定制造业训练数据、森林种子、32 个解释企业和 16 行背景的条件结果；不能直接称为跨行业、跨时期或总体的稳定性证明。

实际完成 50 对（100 个）森林，其中复用 16 个，新增 84 个，失败 0。两机新增计算及回传耗时 {r['seconds']:.1f} 秒；任务分布 {r['by_host']}。训练原始样本为 56,458 条、2,577 公司组、177 个独立首次目标事件。bootstrap 的重复抽样不增加原始独立样本量。模型参数与评价样本均未调整。

## 主要比较：两次独立重训之间的解释变化

{table(p)}

归一化 RMSE 是预先指定的主要指标：将每个企业的贡献向量除以其绝对值之和，减轻仅因 q2 贡献幅度较小而显得稳定的问题。其平均值相对黑箱变化为 {(primary.q2_mean/primary.blackbox_mean-1)*100:+.1f}%。其他指标的区间未做多重比较校正，为辅助描述。

q2 的平均符号一致率为 {float(p.loc[p.metric=='sign_agreement','q2_mean'].iloc[0]):.1%}，应结合这个绝对水平理解相对改善，不能将‘比黑箱更稳定’等同于‘每家企业、每个变量的方向都可靠’。

差值始终定义为 q2−黑箱。RMSE 越小越稳定，排序、集合与符号相似度越大越稳定。归一化 RMSE 无概率单位，不能说成 ST 风险下降了多少百分点；原始 RMSE 才在 CIF 概率贡献尺度上。

### 为什么不是 1,225 个独立样本

50 个公司组重抽样会形成 1,225 个两两比较，同一个模型出现在 49 对里。区间按“删除一个配对重训结果及其所有关联比较”计算 jackknife 标准误，再用 49 自由度 t 近似；**独立重抽样单位是 50，绝不是 1,225**。这是 bootstrap 稳定性泛函均值的条件 Monte Carlo 估计区间，不是从独立总体数据集重新抽样的总体置信区间，有限样本覆盖率未获证明。

之前的 8 次报告主要给两两比较的中位数；本轮预先冻结以均值 U 统计量估计区间。完整 CSV 同时保留每个模型的中位数、配对差值中位数，不能把旧中位数直接当作新均值比较。

## 与各自原始全样本模型比较

这是另外一个估计对象：每次重训分别离自己的原始模型有多远，再取 q2−黑箱；不是 q2 与黑箱模型之间的解释差异。

{table(a[a.n_replicates==50])}

下面是上述 **50 个单次重训差值的 2.5%–97.5% 分位数范围**，反映每次抽样的波动，不是平均差值的置信区间。50 次时尾部分位数本身仍不精确：

| 指标 | 重训差值分布区间 |
|---|---|
'''
    for _,x in a[a.n_replicates==50].iterrows():report+=f'| {LABELS[x.metric]} | [{x.distribution_low:+.6f}, {x.distribution_high:+.6f}] |\n'
    report+='\n## 固定背景与更换背景必须区分\n\n主分析固定背景，仅改变训练公司样本。以下敏感性分析固定原始模型，只更换背景（继承旧 worker 的按公司组抽取 16 行规则），并不重训第二组模型：\n\n'+table(bg[bg.n_replicates==50])
    report+=f'''
不要因为主分析较稳定，就宣称对任意背景分布也更稳定。背景抽样中重复抽中的公司使用既有克隆组标签，结论只覆盖该协议。

## q2 的原生函数与共同 Shapley 比较不是同一件事

两模型比较的是**同一联合行背景、同一 CIF 输出上的精确 Shapley 特征贡献**，因此不存在给黑箱使用低预算近似 SHAP 而给 q2 使用精确解释的不公平。这里不是宣称黑箱拥有简单的完整二阶函数表示。

q2 的完整原生二阶分解以原训练样本的乘积参考分布居中，与共同联合行背景 Shapley 的归因定义不同；另行保存并核验：50 个模型最大重构误差 {r['max_q2_reconstruction_error']:.3g}；两类模型精确 Shapley 最大加回误差 {r['max_shap_additivity_error']:.3g}；保存重载最大误差 {r['max_reload_error']:.3g}。

q2 两次重训的原生函数集合 Jaccard 中位数 {native.function_jaccard.median():.4f}，二元交互集合 Jaccard 中位数 {native.pair_jaccard.median():.4f}，共同非恒定函数在 32 企业上的形状相关性中位数 {native.median_shape_correlation.median():.4f}。函数集合指完整编译函数键，可能包含在这 32 个企业上为零的项。这些仍是依赖的 1,225 对描述统计；不是交互恢复率、因果解释或全输入空间的函数一致性证明。尤其不能把较好的变量归因稳定性转述为所有交互函数都稳定。

## 能支持与不能支持的结论

本轮增加了既定 q2 整体配置相对同后端黑箱的条件归因稳定性证据，并提供区间。两配置同时涉及阶数限制、共享切点与字段覆盖，因此不能把差值单独归因于 q=2，也不能归因于可辨识性惩罚（本森林没有使用该惩罚）。黑箱不是冻结 M2。

未新增 EBM 调优、预测性能搜索或结构恢复模拟，旧基线过拟合和真实交互幅度稀释问题仍然存在；本轮不证明预测非劣、理论可辨识性或方法创新已成功。整体方法创新研究判断仍为 **INCONCLUSIVE**。数据口径仍为 **PROVISIONAL、2023/2024 事后探索性**。

## 可复核产物

- `stability_intervals.csv`：8/30/50 次固定前缀的条件区间与均值、中位数；50 次为终点，未按结果停在 30 次。
- `paired_stability.csv`：全部 1,225 对的两模型指标与配对差值。
- `original_anchor_intervals.csv` / `background_intervals.csv`：原模型距离与背景敏感性。
- `model_hashes.csv` / `task_diagnostics.csv` / `execution_receipt.json`：100 个模型哈希、耗时、内存与数值检查。
- [冻结协议与公式来源](PROTOCOL_ZH.md)。私有模型、企业、预测及解释数组仅在服务器保留。
'''
    (out/'STABILITY_RESULTS_ZH.md').write_text(report)
    fig,axs=plt.subplots(3,2,figsize=(11,8),constrained_layout=True)
    for ax,(_,x) in zip(axs.flat,p.iterrows()):
        ax.axvline(0,color='#777777',ls='--',lw=1)
        ax.errorbar(x.estimate,0,xerr=[[x.estimate-x.ci_low],[x.ci_high-x.estimate]],fmt='o',color='#2471a3',capsize=5)
        ax.set_title(ENGLISH[x.metric],fontsize=11);ax.set_yticks([]);ax.set_ylim(-1,1)
        ax.set_xlabel('q2 minus black box; '+('negative favors q2' if x.metric in ['rmse','normalized_rmse'] else 'positive favors q2'),fontsize=9)
        ax.text(.02,.90,f'{x.estimate:+.5f} [{x.ci_low:+.5f}, {x.ci_high:+.5f}]',transform=ax.transAxes,fontsize=9)
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('50 paired company bootstraps: conditional stability difference\n95% jackknife-t Monte Carlo intervals; fixed 32 anchors / 16 background rows',fontsize=13)
    fig.savefig(out/'stability_intervals.png',dpi=180);fig.savefig(out/'stability_intervals.pdf');plt.close(fig)
    (out/'README.md').write_text('''# 50 次配对 bootstrap 稳定性扩展

[结果与解释](STABILITY_RESULTS_ZH.md) · [冻结协议](PROTOCOL_ZH.md) · [指标区间](stability_intervals.csv) · [执行回执](execution_receipt.json)

固定制造业样本与原 q2/黑箱配置；复用 8 对，新增 42 对，共 50 对。原始企业、参数与解释数组不发布。所有结果为 PROVISIONAL、事后探索性。

![配对差值及条件区间](stability_intervals.png)

复现：`python -m experiments.bootstrap_stability_50_v1.prepare --run UNIQUE_RUN --parent OLD_RUN`；完整测试后提交代码，再 `prepare --run UNIQUE_RUN --repo REPO --freeze`；`hetero-run UNIQUE_RUN/formal.py` 使用两机独立任务；完成后 `python -m experiments.bootstrap_stability_50_v1.evaluate --run UNIQUE_RUN --job JOB_ID`、`python -m experiments.bootstrap_stability_50_v1.report --run UNIQUE_RUN`。环境与原基线一致，见 environment.json / requirements.lock。需要私有服务器输入，仓库汇总不能还原企业明细。
''')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
