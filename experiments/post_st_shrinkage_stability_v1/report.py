"""Aggregate-only scientific figures and compact presentation tables."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .common import *

def main():
    out=D/'figures';out.mkdir(exist_ok=True)
    p=pd.read_csv(D/'PREDICTION_STABILITY.csv')
    f=pd.read_csv(D/'FUNCTION_STABILITY.csv')
    m=pd.read_csv(D/'METRIC_DISTRIBUTIONS.csv')
    delta=pd.read_csv(D/'PAIRED_PERFORMANCE_DIFFERENCES.csv')
    colors={0:'#4c668c',10:'#008575',18:'#c36d30'}
    fig,axs=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    panels=[(p.query('months==12 and cause==1'),'mean_company_sd',100,'Probability SD (percentage points)',False),
        (p.query('months==12 and cause==1'),'normalized_prediction_sd',1,'SD / between-company dispersion',False),
        (p.query('months==12 and cause==1'),'median_pair_rank_correlation',1,'Median rank correlation',True),
        (p.query('months==12 and cause==1'),'top20_jaccard_mean',1,'Mean Top-20% Jaccard',True),
        (f.query('day==365 and cause==1 and order==2'),'mean_common_shape_correlation',1,'Common pair-function correlation',True),
        (f.query('day==365 and cause==1 and order==2'),'mean_normalized_function_difference',1,'Relative pair-function difference',False)]
    for ax,(df,metric,scale,title,up) in zip(axs.flat,panels):
        for si,s in enumerate(STRENGTHS):
            z=df[df.strength==s].sort_values('year')
            ax.bar(np.arange(3)+(si-1)*.24,z[metric]*scale,width=.22,color=colors[s],label=f'Strength {s}')
        ax.set_xticks(range(3),list(YEARS));ax.set_title(title+(' (higher better)' if up else ' (lower better)'),fontsize=10)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('50 paired training-company perturbations per year | termination, 12 months\nAbsolute SD alone is not evidence of improved ranking or explanation stability',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(out/f'stability_summary.{ext}',dpi=200,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.6),constrained_layout=True)
    z=delta[(delta.scope=='pooled')&(delta.months==12)&(delta.cause==1)]
    for ax,metric,title in zip(axs,['within_year_auc','ap','ipcw_brier'],['Within-year AUC difference','Pooled AP difference','Brier difference']):
        for i,contrast in enumerate(['10_minus_0','18_minus_0','18_minus_10']):
            r=z[(z.metric==metric)&(z.contrast==contrast)].iloc[0]
            ax.plot([r.p05,r.p95],[i,i],color='#4c668c',lw=3)
            ax.plot(r['mean'],i,'o',color='#008575')
        ax.axvline(0,color='.5',lw=1,ls='--');ax.set_yticks(range(3),['10 - 0','18 - 0','18 - 10']);ax.set_title(title,fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.tick_params(axis='x',labelsize=9)
        if metric=='ipcw_brier':ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    fig.suptitle('Mean paired difference and P5-P95 training-perturbation range\nDescriptive range, not a confidence interval',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(out/f'performance_tradeoffs.{ext}',dpi=200,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    # Reader-facing compact table; full records stay in their source CSVs.
    z=m[(m.scope=='pooled')&(m.months==12)&(m.cause==1)]
    z.pivot(index='strength',columns='metric',values='mean').reset_index().to_csv(D/'PRIMARY_SUMMARY.csv',index=False)
    p.query('months==12 and cause==1').to_csv(D/'PRIMARY_STABILITY.csv',index=False)
    print('Figures and compact tables complete')
if __name__=='__main__':main()
