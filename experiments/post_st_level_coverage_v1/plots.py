"""Aggregate scientific plots, PNG and PDF. No enterprise identities/parameters."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.post_st_structure_recovery_v1.plots import setup,save,LABELS
from experiments.post_st_structure_recovery_v1.common import SCENARIOS

def plots(run):
    setup();out=run/'reports';colors={'reference':'#2166ac','affine':'#3b927a','wide83':'#c65b32'}
    names={'reference':'当前 q2 · 每树25列','affine':'仅历史仿射校准','wide83':'q2 · 每树83列'}
    df=pd.read_csv(out/'enterprise_metrics.csv',dtype={'scope':str,'seed':str})
    df=df[(df.seed=='ensemble')&(df.cause=='termination')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,h in zip(axes,[24,12]):
        for j,arm in enumerate(colors):
            g=df[(df.horizon_months==h)&(df.arm==arm)].set_index('scope').loc[['2023','2024']]
            ax.bar(np.arange(2)+(j-1)*.22,g.risk_mean*100,width=.2,label=names[arm],color=colors[arm])
        obs=df[(df.horizon_months==h)&(df.arm=='reference')].set_index('scope').loc[['2023','2024']]
        ax.scatter(np.arange(2),obs.ipcw_observed_incidence*100,marker='_',s=850,color='black',label='观察发生率',zorder=4)
        ax.set(xticks=np.arange(2),xticklabels=['2023','2024'],ylabel='先最终退市概率（%）',title=f'{h}个月：平均风险与观察发生率',ylim=(0,55));ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left');fig.tight_layout()
    fig.text(.08,-.04,'校准与扩大候选资格分别实施；未组合。2023/2024已暴露，为探索性比较。',fontsize=9)
    save(fig,out,'enterprise_probability_levels')
    ss=pd.read_csv(out/'simulation_signal_summary.csv',dtype={'seed':str})
    ss=ss[(ss.seed=='ensemble')&(ss.day==730)&(ss.cause==1)&(ss.role=='true_pair')&(ss.metric=='amplitude_ratio')]
    cases=[s for s in SCENARIOS if s!='additive_small'];y=np.arange(len(cases))
    fig,ax=plt.subplots(figsize=(11,6))
    for arm,off in [('reference',-.13),('wide83',.13)]:
        g=ss[(ss.arm==arm)&(ss.variant=='shrunk')].set_index('scenario').loc[cases]
        ax.errorbar(g['mean']*100,y+off,xerr=np.vstack([(g['mean']-g.low95)*100,(g.high95-g['mean'])*100]),fmt='o',capsize=3,color=colors[arm],label=names[arm])
    ax.axvline(100,color='#777777',ls='--',lw=1,label='真实幅度100%');ax.axvline(0,color='#bbbbbb',lw=.6)
    ax.set(yticks=y,yticklabels=[LABELS[s] for s in cases],xlabel='真实二阶交互幅度恢复比例（%）',title='保留q2，扩大每棵树候选资格后的幅度恢复');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=9)
    fig.text(.1,-.01,'50份共同数据的均值和均值95%区间；每份先平均3个种子。连续情形是投影幅度，另报告完整L2误差。',fontsize=9)
    save(fig,out,'interaction_amplitude_recovery')
    ms=pd.read_csv(out/'simulation_metric_summary.csv',dtype={'seed':str});ms=ms[(ms.seed=='ensemble')&(ms.variant=='shrunk')&(ms.day==730)&(ms.cause==1)]
    fig,axes=plt.subplots(1,2,figsize=(12,6));cases=list(SCENARIOS);yy=np.arange(len(cases))
    for ax,metric,title,scale in [(axes[0],'joint_risk_mse','真实CIF均方误差 ×10⁴',1e4),(axes[1],'spatial_cells_including_missing','完整函数表空间单元数',1)]:
        for arm,off in [('reference',-.12),('wide83',.12)]:
            g=ms[(ms.arm==arm)&(ms.metric==metric)].set_index('scenario').loc[cases]
            ax.errorbar(g['mean']*scale,yy+off,xerr=np.vstack([(g['mean']-g.low95)*scale,(g.high95-g['mean'])*scale]),fmt='o',capsize=2,color=colors[arm],label=names[arm])
        ax.set(yticks=yy,yticklabels=[LABELS[s] for s in cases],xlabel=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout(w_pad=3);save(fig,out,'simulation_prediction_complexity')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);plots(p.parse_args().run)
