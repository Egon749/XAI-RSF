"""All figures read audit output tables. Real firm case charts stay private."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from .common import *
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white'})
BLUE='#225d87'; RED='#ad4d37'; GREEN='#218074'; GREY='#8c9aa3'
def data(name):return pd.read_csv(HERE/(name+'.csv'))
def save(fig,name,out):
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(out/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)
def figures():
    out=HERE/'figures';out.mkdir(exist_ok=True)
    df=data('ALL_METRICS').query("version=='shrunk' and cause==1");df['scope']=df.scope.astype(str)
    cx=data('COMPLEXITY_TRADEOFF').query("version=='shrunk'");cx['model']=cx.model.astype(str)
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    periods=[('2023',12),('2023',24),('2024',12),('2024',24),('2025',12)]
    for ax,m in zip(axes,['auc','ap']):
        for i,(y,h) in enumerate(periods):
            d=df[(df.scope==y)&(df.months==h)].set_index('model')
            z=d.loc[NAMES,m]
            ax.boxplot(z,positions=[i],orientation='horizontal',widths=.45,showfliers=True,whis=(5,95),patch_artist=True,boxprops={'facecolor':'#d6e8ef'},medianprops={'color':BLUE})
            for label,color,mk in [('reference96',RED,'D'),('ensemble320',GREEN,'o')]:
                ax.scatter(d.at[label,m],i+.12,color=color,s=25,marker=mk,zorder=4)
        ax.set_xlabel(m.upper());ax.set_yticks(range(5),[f'{y} / {h}m' for y,h in periods]);ax.grid(axis='x',alpha=.18)
    axes[0].invert_yaxis();fig.suptitle('Termination performance across the 120 fixed triples',y=1.07)
    fig.legend(handles=[Line2D([],[],marker='D',ls='',color=RED,label='Original 96'),Line2D([],[],marker='o',ls='',color=GREEN,label='320 trees')],loc='upper center',bbox_to_anchor=(.5,1.015),ncols=2)
    save(fig,'01_performance',out)
    fig,axes=plt.subplots(1,3,figsize=(10,3),sharey=True)
    for ax,y in zip(axes,YEARS):
        d=df[(df.scope==str(y))&(df.months==12)].merge(cx[cx.year==y],on='model')
        for title,sel,col,mk in [('Singles',d.model.isin(list(map(str,SEEDS))),GREY,'x'),('Triples',d.model.isin(NAMES),BLUE,'.'),('Original 96',d.model=='reference96',RED,'D'),('320 trees',d.model=='ensemble320',GREEN,'o')]:
            ax.scatter(d[sel].interaction_functions,d[sel].auc,s=23,marker=mk,color=col,label=title,alpha=.8)
        ax.set_title(str(y));ax.set_xlabel('Nonzero interaction functions')
    axes[0].set_ylabel('12m termination AUC');axes[1].legend(bbox_to_anchor=(.5,-.28),loc='upper center',ncols=4)
    save(fig,'02_prediction_complexity',out)
    for o,filename in [(1,'MAIN_EFFECT_STABILITY'),(2,'INTERACTION_STABILITY')]:
        t=data(filename).query("version=='shrunk' and cause==1 and day==365 and region=='observed' and comparison=='disjoint'")
        fig,axes=plt.subplots(1,3,figsize=(10,3.4),sharex=True,sharey=True)
        for y,ax in zip(YEARS,axes):
            g=t[t.year==y]
            art=ax.scatter(g.presence,g.shape_correlation,c=np.log10(np.maximum(g.mean_abs_median*100,1e-4)),cmap='viridis',vmin=-4,vmax=.5,s=17,alpha=.75)
            ax.axvline(.9,ls=':',color=GREY);ax.axhline(.8,ls=':',color=GREY);ax.set_title(str(y));ax.set_xlabel('Fraction of triples containing term')
        axes[0].set_ylabel('Shape correlation, disjoint triples');axes[0].set_ylim(-1.06,1.08)
        cb=fig.colorbar(art,ax=axes[-1],fraction=.07);cb.set_label('log10 mean |CIF points| (pp)')
        save(fig,'03_main_stability' if o==1 else '04_interaction_stability',out)
    curves=data('MAIN_CURVE_BANDS').query("version=='shrunk' and day==365 and cause==1 and not missing")
    fig,axes=plt.subplots(1,3,figsize=(10,3.1))
    for ax,key in zip(axes,['prior_audit_standard','prior_audit_going_concern','max_drawdown_1m']):
        for y,color in zip(YEARS,[BLUE,GREEN,RED]):
            d=curves[(curves.term==key)&(curves.year==y)].sort_values('bin')
            # Each point corresponds to a supported empirical cell, no interpolation of model cuts.
            x=d.bin.to_numpy() if key!='max_drawdown_1m' else 100*d.percentile_midpoint.to_numpy()
            ax.errorbar(x+(.07*(y-2024) if key!='max_drawdown_1m' else 0),100*d['median'],yerr=100*np.stack([d['median']-d.p05,d.p95-d['median']]),fmt='o',ms=4,capsize=2,color=color,label=str(y))
        ax.axhline(0,ls=':',color=GREY);ax.set_title(key.replace('prior_audit_','Audit: ').replace('_',' '),fontsize=9)
        ax.set_xlabel('Training percentile (cell midpoint)' if key=='max_drawdown_1m' else 'Indicator state (0 or 1)')
        if key!='max_drawdown_1m':ax.set_xticks([0,1])
    axes[0].set_ylabel('12m termination contribution (pp)');axes[0].legend()
    save(fig,'05_function_bands',out)
    d=data('LOCAL_PROFILE_GROUPS').query("year==2024 and months==12 and cause==1 and version=='shrunk'")
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),sharex=True)
    for ax,band in zip(axes,['stable','unstable']):
        p=d[d.band==band].nlargest(5,'mean_absolute').iloc[::-1]
        ax.barh(p.term.str.replace('prior_audit_','audit: '),100*p.mean_absolute,color=BLUE if band=='stable' else RED)
        ax.set_title(f'{"Lowest" if band=="stable" else "Highest"} L1 quintile (n={int(p.companies.iloc[0])})')
        ax.set_xlabel('Mean absolute contribution (pp)')
    save(fig,'06_local_profiles',out)
    # Complete real individual reports are restricted to the private run.
    r=run_path();private=r/'case_figures_PRIVATE';private.mkdir(exist_ok=True)
    for z in read(r/'explain/published.json'):
        y,v=z['args']
        if v!='shrunk':continue
        p=Path(z['path'])
        for h in (12,24):
            for q in (.1,.9):
                item=arrays(p/f'case_{h}_1_{q}_PRIVATE.npz');x=item['contributions']*100
                names=[f['feature_name'] for f in read(r/'input/features.json')]
                keys=[' × '.join(names[int(j)] for j in str(k).split('+')) for k in item['terms']]
                idx=np.argsort(-abs(x).mean(0))[:10][::-1];mid=np.median(x[:,idx],0);lo,hi=np.quantile(x[:,idx],[.05,.95],axis=0)
                fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(mid,np.arange(len(idx)),xerr=[mid-lo,hi-mid],fmt='o',color=BLUE)
                ax.set_yticks(np.arange(len(idx)),[keys[j] for j in idx]);ax.axvline(0,color=GREY);ax.set_xlabel('Signed contribution (pp); median and 5–95% across 120 triples')
                ax.set_title(f'Private fixed case | {y}, {h}m | L1 quantile {q:g}')
                save(fig,f'{y}_{h}_{q}_PRIVATE',private)
if __name__=='__main__':figures()
