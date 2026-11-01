"""Aggregate scientific plots; no company data or fitted thresholds exported."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from .common import load

def draw(run):
    report=run/'reports';out=report/'figures';out.mkdir(exist_ok=True)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    d=pd.read_csv(report/'DROP_COLUMN_ALL_83.csv');fs=load(run/'input/features.json')
    # Ranking is descriptive within-year AP, never a retrospective feature decision.
    r=d[(d.horizon==12)&(d.cause==1)&d.mature&d.supported&(d.known_termination_events>=1)&(d.validation_target_events>=3)]
    m=r.groupby(['outer_cutoff','feature_id']).delta_ap.mean().unstack(0).reindex(range(83))
    order=m[2025].sort_values().index;fig,ax=plt.subplots(1,3,figsize=(17,24),sharey=True)
    yy=np.arange(83)
    for i,y in enumerate([2023,2024,2025]):
        vals=m.loc[order,y].to_numpy();ax[i].barh(yy,vals,color=np.where(vals>0,'#c66346','#3274a1'),height=.72);ax[i].axvline(0,color='#777',lw=.8)
        ax[i].set_title(f'{y}年截止前内部验证');ax[i].set_xlabel('删除后 AP − 全83项 AP');ax[i].grid(axis='x',alpha=.2);ax[i].set_axisbelow(True)
    ax[0].set_yticks(yy,[f'{j:02d}  {fs[j]["label"]}' for j in order],fontsize=8)
    fig.suptitle('全部83项逐一删除重训：12个月先退市AP变化\n仅有效历史队列等权均值；正值表示删除有利，事件支持不足以批准自动删除',y=.998)
    fig.tight_layout(rect=(0,0,1,.98));save(fig,'drop_column_all83_ranking')
    internal=pd.read_csv(report/'INTERNAL_MODEL_COMPARISON.csv',dtype={'year':str,'seed':str,'cohorts':str});cx=pd.read_csv(report/'INTERNAL_EXPLANATION_COMPLEXITY.csv')
    q=internal[(internal.scope=='inner_mature_pooled')&(internal.seed=='ensemble')&(internal.cause==1)]
    for h in [12,24]:
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
        for col,y in enumerate([2023,2024,2025]):
            z=q[(q.outer_cutoff==y)&(q.horizon==h)]
            for row,metric in enumerate(['ap','auc']):
                ax=axes[row,col];ax.scatter(z.features,z[metric],s=17,alpha=.6,c=np.where(z['mode'].eq('ratio'),'#d88541','#3274a1'))
                ax.set_ylabel(metric.upper());ax.grid(alpha=.2);ax.set_title(f'{y}截止 / {h}个月');ax.set_xlim(33,85)
                ax.axvspan(35,65,color='#eeeeee');ax.set_xticks([35,45,55,65,74,83]);ax.set_xlabel('输入变量数')
        fig.suptitle('历史内部合并预测的实测候选；灰区未运行，未强迫删除至较小规模');fig.tight_layout();save(fig,f'feature_count_performance_{h}m')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,y in zip(axes,[2023,2024,2025]):
        z=cx[cx.year<y].groupby(['subset_id','features','mode']).interaction_functions.mean().reset_index()
        ax.scatter(z.features,z.interaction_functions,s=18,alpha=.6);ax.set_title(f'{y}截止前森林');ax.set_xlabel('输入变量数');ax.set_ylabel('96树预测器非零交互数均值');ax.grid(alpha=.2)
    fig.tight_layout();save(fig,'feature_count_interaction_complexity')
    freq=pd.read_csv(report/'FEATURE_SELECTION_STABILITY.csv');freq=freq.pivot(index='feature_id',columns='outer_cutoff',values='main_nonzero_frequency_12m_termination').reindex(range(83))
    fig,axes=plt.subplots(1,2,figsize=(11,22),sharey=True)
    axes[0].imshow(np.ones((83,3)),cmap=matplotlib.colors.ListedColormap(['#dddddd']),aspect='auto',vmin=0,vmax=1)
    axes[0].set_title('自动特征选择频率：NA\n三年均因历史事件不足回退')
    image=axes[1].imshow(freq.to_numpy(),cmap='Blues',aspect='auto',vmin=0,vmax=1);axes[1].set_title('回退模型12个月退市非零主效应频率\n50次公司重抽样，同一历史参考样本')
    for ax in axes:ax.set_xticks(range(3),['2023','2024','2025'])
    axes[0].set_yticks(range(83),[f'{j:02d} {fs[j]["label"]}' for j in range(83)],fontsize=8)
    fig.colorbar(image,ax=axes[1],fraction=.03,pad=.04,label='函数出现频率（不是变量选择频率）');fig.subplots_adjust(left=.32,wspace=.25);save(fig,'feature_selection_stability_heatmap')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,y in zip(axes,[2023,2024,2025]):
        z=q[(q.outer_cutoff==y)&(q.horizon==12)].copy()
        z['cells']=[cx[(cx.subset_id==r.subset_id)&cx.year.isin([int(v) for v in r.cohorts.split(',')])].spatial_table_cells.mean() for r in z.itertuples()]
        dots=ax.scatter(z.cells,z.ap,c=z.features,cmap='viridis',s=22,alpha=.7,vmin=q.features.min(),vmax=q.features.max())
        fig.colorbar(dots,ax=ax,fraction=.035,pad=.03,label='输入变量数')
        vals=z.sort_values(['cells','ap'],ascending=[True,False]);front=[];best=-np.inf
        for row in vals.itertuples():
            if row.ap>best:front.append((row.cells,row.ap));best=row.ap
        if front:ax.plot(*zip(*front),color='#c66346',lw=1)
        ax.set_title(f'{y}截止 / 历史12个月');ax.set_xlabel('完整空间单元数（同组验证年份均值）');ax.set_ylabel('合并 AP');ax.grid(alpha=.2)
    fig.suptitle('实测预测—解释复杂度前沿；描述性候选边界，不是已证明性能等效');fig.tight_layout();save(fig,'prediction_complexity_pareto')
    outer=pd.read_csv(report/'MODEL_COMPARISON.csv');z=outer[(outer.seed=='ensemble')&(outer.cause==1)&outer.year.isin(['2023','2024','2025'])&outer.label.isin(['full83','legacy74','automatic_fallback83'])&(outer['mode']=='count')]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    labels=['full83','legacy74','automatic_fallback83'];names=['83变量','74变量','自动筛选回退83']
    for row,h in enumerate([12,24]):
        for col,key in enumerate(['auc','ap','brier']):
            ax=axes[row,col]
            for i,(label,name) in enumerate(zip(labels,names)):
                g=z[(z.horizon==h)&(z.label==label)].set_index('year').reindex(['2023','2024','2025'])
                ax.bar(np.arange(3)+(i-1)*.24,g[key],width=.23,label=name)
            ax.set_xticks(range(3),['2023','2024','2025']);ax.set_title(f'{h}个月 {key.upper()}'+('（越低越好）' if key=='brier' else ''));ax.grid(axis='y',alpha=.2)
            if row==1:ax.text(.98,.96,'2025：随访不足，NA',ha='right',va='top',transform=ax.transAxes,fontsize=8)
    axes[0,0].legend(fontsize=8);fig.suptitle('已暴露外层年度：83、74与自动回退结果；回退使用同一83列预测');fig.tight_layout();save(fig,'annual83_74_automatic_comparison')
    print('figures',len(list(out.glob('*.png'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();draw(a.run)
