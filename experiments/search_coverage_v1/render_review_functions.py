"""Render every stable main effect from the released percentile-coordinate JSON.

Only standard-library/NumPy/Matplotlib; no private files and no fitting.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    'loss_flag': '亏损标记', 'interest_coverage': '利息保障倍数', 'roa': 'ROA',
    'net_profit_to_assets': '净利润/总资产', 'operating_profit_to_assets': '营业利润/总资产',
    'gross_margin': '毛利率', 'net_margin': '净利率',
    'financial_expense_to_revenue': '财务费用/营业收入', 'net_debt_to_assets': '净债务/总资产',
    'leverage_change': '杠杆变化', 'total_assets_growth': '总资产增长',
    'ret_cum_1m': '近1个月累计收益率', 'max_drawdown_1m': '近1个月最大回撤',
    'ret_vol_1m': '近1个月收益波动率', 'log_amihud_illiq_1m': '近1个月Amihud非流动性对数',
    'operating_margin': '营业利润率', 'market_missing_coverage': '市场数据缺失覆盖',
}


def save_svg(fig, path):
    fig.savefig(path, metadata={'Date': None})
    # Matplotlib emits trailing spaces in SVG path data; retain separators as newlines.
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip()+'\n')


def main(directory, previews=None):
    source = directory/'stable_functions_percentile.json'
    data = json.loads(source.read_text())
    figures = directory/'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'svg.fonttype':'none','svg.hashsalt':'xai-rsf-fixed-function-release',
                         'axes.spines.top':False,'axes.spines.right':False})
    colors = {42:'#2465a6',101:'#cc7625',202:'#758441'}
    styles = {42:'-',101:'--',202:':'}
    titles = ['3 months (90 days, approx.)','6 months (180 days, approx.)','12-month display (365 days)']
    docs = ['# 稳定入选主效应：全部15个函数形状', '',
            '稳定指V19/V20三个种子共六模型都选择了该变量，不代表形状稳定。', '',
            '每图上排V19、下排V20；每列对应90/180/365日，三条线为种子42/101/202。纵轴为目标ST的CIF贡献（百分点），不是风险率或因果影响。', '',
            '横轴为各训练折非缺失输入的经验分位位置，同一百分位在两折不一定对应相同原始值；NA为单独的缺失状态，未接入数值曲线。每个函数的六面板共用纵轴尺度，不同函数尺度不同。没有平滑或补出新的切点。', '',
            '90/180日仅为3/6个月近似展示；原始研究的12个日历月评价口径未改。365/366日数值均在JSON和CSV内。所有区间精确函数值见[JSON](stable_functions_percentile.json)。这15个主效应不是完整森林全部解释项。', '']
    assets=[]
    for name in data['stable_main_names']:
        fig,axes=plt.subplots(2,3,figsize=(14,8.4),sharey=True)
        allvals=[]
        observedvals=[]
        missingvals=[]
        for row,fold in enumerate(['V19','V20']):
            for model in [m for m in data['models'] if m['fold']==fold]:
                fun=next(f for f in model['functions'] if f['feature_names']==[name])
                values=np.asarray(fun['values'])*100
                allvals.extend(values[:,:3].ravel())
                observedvals.extend(values[:-1,:3].ravel())
                missingvals.extend(values[-1,:3].ravel())
                edges=np.r_[0.,fun['cut_percentiles'][0],100.]
                assert len(edges)==len(values)
                assert np.all(np.diff(edges)>=0)
                seed=model['seed']
                for col in range(3):
                    ax=axes[row,col]
                    ax.stairs(values[:-1,col],edges,baseline=None,color=colors[seed],
                              linestyle=styles[seed],linewidth=2,label=f'Seed {seed}')
                    ax.scatter([110],[values[-1,col]],color=colors[seed],s=34,
                               marker={42:'o',101:'s',202:'^'}[seed],zorder=3)
                    ax.set_title(f'{fold} | {titles[col]}',fontsize=11)
        low,high=min(0.,min(allvals)),max(0.,max(allvals))
        pad=max((high-low)*.12,.01)
        for row in range(2):
            for col in range(3):
                ax=axes[row,col]
                ax.set_xlim(-2,115);ax.set_ylim(low-pad,high+pad)
                ax.set_xticks([0,25,50,75,100,110],['0','25','50','75','100','NA'])
                ax.axhline(0,color='#555555',linewidth=.7)
                ax.axvline(105,color='#b0b0b0',linewidth=.7,linestyle=':')
                ax.grid(axis='y',color='#dddddd',linewidth=.6)
                if row==1:ax.set_xlabel('Training percentile (observed values)')
                if col==0:ax.set_ylabel('CIF contribution (percentage points)')
        fig.suptitle(name,fontsize=17,y=.985)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
        fig.text(.5,.012,'Frozen coverage models | Target ST | NA is a separate missing state | No confidence intervals',ha='center',fontsize=10)
        fig.subplots_adjust(top=.86,bottom=.10,left=.08,right=.98,hspace=.32,wspace=.18)
        path=figures/f'{name}.svg'
        save_svg(fig,path)
        if previews is not None and name in ['loss_flag','financial_expense_to_revenue','max_drawdown_1m']:
            previews.mkdir(parents=True,exist_ok=True)
            fig.savefig(previews/f'{name}.png',dpi=120)
        docs.extend([f'## {LABELS.get(name,name)}：`{name}`','',f'![{name}](figures/{name}.svg)',''])
        assets.append(dict(feature=name,file=str(path.relative_to(directory)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        # Separate extra view preserves the full plot; never suppress large NA effects.
        if max(abs(v) for v in missingvals) > 5*max(max(abs(v) for v in observedvals),1e-12):
            lower,upper=min(0.,min(observedvals)),max(0.,max(observedvals))
            margin=max((upper-lower)*.12,.001)
            for ax in axes.ravel():
                ax.set_xlim(-2,102)
                ax.set_xticks([0,25,50,75,100],['0','25','50','75','100'])
                ax.set_ylim(lower-margin,upper+margin)
            fig.suptitle(name+' | observed-value detail',fontsize=17,y=.985)
            fig.texts[-1].set_text('Observed bins only; NA excluded from this detail view. See the full plot for missing-state contributions.')
            detail=figures/f'{name}__observed_only.svg'
            save_svg(fig,detail)
            if previews is not None and name=='max_drawdown_1m':
                fig.savefig(previews/f'{name}__observed_only.png',dpi=120)
            assets.append(dict(feature=name,file=str(detail.relative_to(directory)),view='observed_only_detail',sha256=hashlib.sha256(detail.read_bytes()).hexdigest()))
            docs.extend(['缺失点幅度较大，下面另给非缺失区间的放大图；缺失效应仍完整保留在上图。','',f'![{name} observed detail](figures/{name}__observed_only.svg)',''])
        plt.close(fig)
    (directory/'MAIN_EFFECTS_ZH.md').write_text('\n'.join(docs).rstrip()+'\n')
    rows=list(csv.DictReader((directory/'stable_interaction_shares.csv').open()))
    ranked=sorted([r for r in rows if r['scope']=='six_models_equal' and r['days']=='365'],
                  key=lambda r:float(r['share_of_interaction_absolute_sum_mean']),reverse=True)
    report=['# 稳定入选交互及贡献占比','',
            '这里稳定仅指全部六模型都选择；不证明函数方向稳定或统计可辨识性。', '',
            '每折固定512条训练参考记录。交互强度为mean(abs(f_jk))；主要占比分母为该模型全部交互的绝对贡献均值之和，随后对模型等权平均。不是8个稳定交互内部重新归一化。第二分母为全部主效应与交互绝对贡献之和，均不含基准。份额不是预测风险比例，也不是删除该交互的损失。', '',
            '以下为365日目标ST；90、180、366日及全部逐种子值见CSV。年度范围为三个种子的最小—最大值，不是置信区间。', '',
            '|变量对|六模型平均交互份额|V19均值〔种子范围〕|V20均值〔种子范围〕|占全部函数绝对贡献均值|',
            '|---|---:|---:|---:|---:|']
    for r in ranked:
        cells=[]
        for fold in ['V19','V20']:
            f=next(v for v in rows if v['scope']==fold and v['days']=='365' and v['feature1']==r['feature1'] and v['feature2']==r['feature2'])
            vals=[100*float(f['share_of_interaction_absolute_sum_'+s]) for s in ['mean','min','max']]
            cells.append(f'{vals[0]:.3f}%〔{vals[1]:.3f}—{vals[2]:.3f}%〕')
        pair=' × '.join(LABELS.get(r[k],r[k]) for k in ['feature1','feature2'])
        report.append(f"|{pair}|{100*float(r['share_of_interaction_absolute_sum_mean']):.3f}%|{cells[0]}|{cells[1]}|{100*float(r['share_of_all_function_absolute_sum_mean']):.3f}%|")
    report+=['',f"8个共同交互的平均份额合计为{sum(100*float(r['share_of_interaction_absolute_sum_mean']) for r in ranked):.3f}%；其余贡献分布在非六模型共同入选的交互中。",'',
             '[完整稳定交互占比CSV](stable_interaction_shares.csv) · [全部交互逐模型CSV](interaction_importance_by_model.csv)','']
    (directory/'STABLE_INTERACTIONS_ZH.md').write_text('\n'.join(report))
    features=list(csv.DictReader((directory/'feature_importance_summary.csv').open()))
    ranked=sorted([r for r in features if r['scope']=='six_models_equal' and r['days']=='365'],
                  key=lambda r:float(r['main_mean_abs_mean']),reverse=True)
    report=['# 全部75变量的重要性与种子范围','',
            '主排名使用主效应函数在固定训练参考样本上的mean(abs(f_j))。每折512条参考记录，三个种子共用同一批参考记录，未选中函数计零。下表乘100表示CIF百分点尺度的绝对幅度，不是单调风险增量或因果效果。', '',
            '365日目标ST。年度范围为该折三个种子的最小—最大值，不是置信区间；六模型均值是两折等权的描述性汇总。', '',
            'CSV另外列出两种不同的综合指标：mean(abs(f_j + 0.5 sum_k f_jk))，以及mean(abs(f_j)) + 0.5 sum_k mean(abs(f_jk))。前者先带符号分摊再取绝对值，后者为绝对幅度计分，不应混用。这里不是置换重要性、删项重要性或基于预测损失的重要性。', '',
            '此前对全部验证记录的贡献排序采用不同参考样本和综合幅度计分，不能直接与本表数值比较。', '',
            '|排序|变量|入选模型/6|六模型主效应均值×100|V19均值〔种子范围〕×100|V20均值〔种子范围〕×100|',
            '|---:|---|---:|---:|---:|---:|']
    for index,r in enumerate(ranked,1):
        cells=[]
        for fold in ['V19','V20']:
            row=next(v for v in features if v['scope']==fold and v['days']=='365' and v['feature']==r['feature'])
            v=[100*float(row['main_mean_abs_'+s]) for s in ['mean','min','max']]
            cells.append(f'{v[0]:.5f}〔{v[1]:.5f}—{v[2]:.5f}〕')
        label=LABELS.get(r['feature'],'')
        report.append(f"|{index}|{label} `{r['feature']}`|{r['selected_models']}|{100*float(r['main_mean_abs_mean']):.5f}|{cells[0]}|{cells[1]}|")
    report+=['','[全部时点汇总CSV](feature_importance_summary.csv) · [每个种子CSV](feature_importance_by_model.csv)','']
    (directory/'FEATURE_IMPORTANCE_ZH.md').write_text('\n'.join(report))
    (directory/'render_receipt.json').write_text(json.dumps(dict(
        renderer='Matplotlib '+matplotlib.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures=len(assets),panels_per_figure=6,seed_curves_per_panel=3,
        plotted_days=[90,180,365],additional_numeric_day=366,
        y_scale='Shared within each feature, independent across features',assets=assets),indent=2)+'\n')
    print(json.dumps(dict(figures=len(assets),main_effects=len(data['stable_main_names']),stable_pairs=len(data['stable_pairs']))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--previews',type=Path)
    args=p.parse_args()
    main(args.directory,args.previews)
