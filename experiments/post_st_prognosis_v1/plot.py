from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(reports):
    s=pd.read_csv(reports/'metrics_summary.csv')
    s=s[(s.horizon_months==24)&(s.cause=='termination')]
    out=reports/'figures';out.mkdir(exist_ok=False)
    models=['xai_q2','unrestricted','standard_rsf']
    labels=['XAI-RSF q2','Unrestricted forest','Standard RSF']
    colors=['#1976d2','#e08026','#65747e'];scopes=['2023','2024','pooled']
    fig,ax=plt.subplots(1,3,figsize=(12.5,4.2),layout='constrained')
    for axis,metric,title in zip(ax,['auc','ap','brier'],['ROC-AUC (higher is better)','Average precision (higher is better)','Brier score (lower is better)']):
        for j,(m,label,color) in enumerate(zip(models,labels,colors)):
            g=s[s.model.eq(m)].set_index('scope').loc[scopes];mean=g[metric+'_mean'].to_numpy()
            low=mean-g[metric+'_min'].to_numpy();high=g[metric+'_max'].to_numpy()-mean
            axis.errorbar(np.arange(3)+(j-1)*.12,mean,yerr=np.array([low,high]),fmt='o',capsize=3,color=color,label=label)
        axis.set_xticks(range(3),['2023\nn=36, events=11','2024\nn=45, events=20','Pooled\nn=81, events=31'])
        axis.set_title(title,fontsize=11);axis.grid(axis='y',alpha=.2);axis.spines[['top','right']].set_visible(False)
    ax[0].axhline(.5,color='#999999',linestyle=':',linewidth=1)
    ax[0].set_ylim(.40,.75);ax[1].set_ylim(.25,.70);ax[2].set_ylim(.24,.40)
    ax[0].legend(fontsize=8,loc='upper right')
    fig.suptitle('Post-ST first termination within 24 calendar months | announcement clock\nMean and range of 3 seeds (not confidence intervals); provisional retrospective evaluation',fontsize=12)
    for suffix in ['png','svg']:fig.savefig(out/f'prediction_comparison.{suffix}',dpi=190)
    plt.close(fig)
    m=pd.read_csv(reports/'metrics.csv',dtype={'seed':str})
    m=m[(m.seed=='ensemble')&(m.horizon_months==24)&(m.cause=='termination')&m.scope.isin(scopes)]
    fig,ax=plt.subplots(figsize=(7.4,4.3),layout='constrained')
    x=np.arange(3)
    for j,(model,label,color) in enumerate(zip(models,labels,colors)):
        g=m[m.model.eq(model)].set_index('scope').loc[scopes]
        ax.bar(x+(j-1)*.18,g.risk_mean,width=.17,color=color,label=label)
    obs=np.array([11/36,20/45,31/81]);ax.scatter(x,obs,marker='D',s=45,color='#be322d',zorder=5,label='Observed first termination')
    ax.set_xticks(x,['2023','2024','Pooled']);ax.set_ylabel('24-month probability');ax.set_ylim(0,.52)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1));ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,fontsize=8,loc='upper left')
    ax.set_title('All three models substantially underpredict termination\n3-seed ensemble; no test-set recalibration',fontsize=12)
    for suffix in ['png','svg']:fig.savefig(out/f'calibration_in_the_large.{suffix}',dpi=190)
    plt.close(fig)
    # Matplotlib emits trailing spaces in SVG path attributes. Normalize only
    # XML whitespace; keep every coordinate and displayed datum unchanged.
    for path in out.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);plot(p.parse_args().reports)
