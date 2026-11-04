"""Plot aggregate scores only."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .schedule import D, YEARS


def main():
    t = pd.read_csv(D/'ROUND_CURVES.csv')
    null = pd.read_csv(D/'PERMUTATION_RUNS.csv')
    dest = D/'figures'
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(9,8), sharex=True, sharey=True)
    for i,year in enumerate(YEARS):
        g = t[t.year.eq(year)]
        for ax, metric in zip(axes[i], ['ap','auc']):
            ax.plot(g['round'], g['train_'+metric], color='#00756f', lw=2, label='Apparent training')
            ax.plot(g['round'], g['cv_'+metric], color='#aa4d2c', lw=2, label='Three-fold validation')
            ax.set_title(f'{year}  |  {metric.upper()}', loc='left', fontsize=11)
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            ax.set_ylim(-.025,1.045)
    for ax in axes[-1]:
        ax.set_xlabel('Boosting round')
        ax.set_xticks([1,8,16,24,32,40,48])
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,.96),frameon=False)
    fig.suptitle('PROBoost | 12-month delisting, pre-cutoff companies only',weight='bold')
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(dest/'ROUND_CURVES.png',dpi=180)
    plt.close(fig)

    fig,axes = plt.subplots(1,2,figsize=(9,3.9),sharey=True)
    for ax,metric,title in zip(axes,['train_ap','cv_ap'],['Apparent training AP','Three-fold validation AP']):
        for i,year in enumerate(YEARS):
            y = null.loc[null.year.eq(year),metric].dropna()
            q = y.quantile([0,.05,.5,.95,1]).to_numpy()
            ax.hlines(i,q[0],q[4],color='#8b9899',lw=1)
            ax.hlines(i,q[1],q[3],color='#368282',lw=5)
            ax.plot(q[2],i,'o',color='#155c60')
            real = t.loc[t.year.eq(year)&t['round'].eq(48),metric].item()
            ax.plot(real,i-.12,marker='D',color='#a54624',ms=6)
        ax.set_xlim(-.025,1.03)
        ax.set_ylim(2.6,-.6)
        ax.set_yticks(range(3),YEARS)
        ax.set_title(title,loc='left',fontsize=12)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('Original labels and 200 shuffled-label fits per year | round 48',weight='bold')
    fig.text(.07,.06,'Orange diamond: original labels.  Teal dot: shuffled median.  Thick line: P05–P95.  Thin line: full range.',fontsize=9)
    fig.tight_layout(rect=(0,.12,1,.92))
    fig.savefig(dest/'PERMUTATION_AP.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
