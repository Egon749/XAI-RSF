"""Publication figures from aggregate CSVs; no private company information."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

LABEL={'q2':'XAI q2','q3':'XAI q3','q4':'XAI q4','ebm_gam':'EBM-GAM','ebm_ga2m10':'GA2M-10','ebm_ga2m30':'GA2M-30','unrestricted':'Unrestricted RSF','q2_coverage':'q2 coverage'}
COLOR={m:('#2864A0' if m.startswith('q') else '#B97718') for m in LABEL}
MARKER={'q2':'o','q3':'^','q4':'s','ebm_gam':'D','ebm_ga2m10':'v','ebm_ga2m30':'P'}

def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=180,facecolor='white');fig.savefig(out/f'{name}.svg',facecolor='white');plt.close(fig)

def main(source,out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold','svg.fonttype':'none'})
    m=pd.read_csv(source/'metrics.csv');c=pd.read_csv(source/'native_complexity.csv');x=c.groupby('model').spatial_cells.mean()
    models=list(MARKER);source_rows=[]
    fig,axes=plt.subplots(3,2,figsize=(13,13),layout='constrained')
    for row,scope in enumerate(['2023','2024','pooled']):
        d=m[(m.scope==scope)&(m.aggregation=='single_seed')&(m.weighting=='record_equal')]
        for col,metric in enumerate(['auc','ap']):
            ax=axes[row,col];means=d.groupby('model')[metric].mean();front=[]
            for model in models:
                yy=d[d.model==model][metric];xx=x[model];v=means[model]
                ax.errorbar(xx,v,yerr=[[v-yy.min()],[yy.max()-v]],fmt=MARKER[model],color=COLOR[model],ms=8,capsize=3,elinewidth=1.1)
                offset={'q2':(-8,-17),'q3':(8,-17),'q4':(-8,10),'ebm_gam':(8,-17),'ebm_ga2m10':(8,10),'ebm_ga2m30':(8,-17)}[model]
                ax.annotate(LABEL[model],(xx,v),xytext=offset,textcoords='offset points',ha='right' if offset[0]<0 else 'left',fontsize=9,color=COLOR[model])
                dominated=any(x[k]<=xx and means[k]>=v and (x[k]<xx or means[k]>v) for k in models if k!=model)
                if not dominated:front.append((xx,v))
                source_rows.append(dict(scope=scope,metric=metric,model=model,cells=float(xx),mean=float(v),min=float(yy.min()),max=float(yy.max()),descriptive_nondominated=not dominated,native_scale='CIF' if model.startswith('q') else 'log_odds'))
            front.sort()
            if len(front)>1:ax.plot(*np.array(front).T,ls=':',lw=1,color='#777777',zorder=0)
            for ref,style,label in [('unrestricted','--','Unrestricted RSF'),('M2_allsector','-.','M2 all-sector')]:
                ax.axhline(means[ref],ls=style,color='#555555',lw=1,label=label)
            ax.set_xscale('log');ax.set_xlim(650,220000);ax.set_xlabel('Allocated native function-table cells (log scale)');ax.set_ylabel('IPCW '+metric.upper()+' (higher is better)');ax.set_title(f'{scope.capitalize()}: prediction vs explanation size');ax.grid(axis='y',color='#DDDDDD',lw=.6)
            ax.margins(y=.25)
            if row==0 and col==0:ax.legend(loc='best',frameon=False,fontsize=8)
    fig.suptitle('Manufacturing first-ST prediction: descriptive accuracy–complexity frontier',fontsize=15)
    fig.supxlabel('Three-seed means and min–max ranges, not confidence intervals. Blue: CIF functions; gold: log-odds functions + logistic link.\nOnly frozen configurations; EBM fixed-round fits overfit (not tuned-baseline superiority). Previously exposed data, PROVISIONAL.\nReference lines have no claimed global function-table size.',fontsize=9)
    save(fig,out,'accuracy_complexity_frontier');pd.DataFrame(source_rows).to_csv(out/'frontier_source.csv',index=False)
    sim=pd.read_csv(source/'simulation_metrics.csv');scenarios=['additive_cif','pair_cif','triple_cif','additive_logit','pair_logit','correlated_pair','heavy_censor_pair'];methods=['q2_coverage','q2','q3','unrestricted','ebm_gam','ebm_ga2m10']
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for ax,metric,title in zip(axes,['f1','observed_distribution_probability_mse'],['Probability-scale structure recovery (F1, higher better)','True event-probability MSE (lower better)']):
        a=sim.groupby(['scenario','model'])[metric].mean().unstack().reindex(index=scenarios,columns=methods)
        im=ax.imshow(a.to_numpy(),cmap='Blues' if metric=='f1' else 'Blues_r',aspect='auto',vmin=0,vmax=1 if metric=='f1' else None)
        ax.set_xticks(range(len(methods)),[('q2 all-features' if k=='q2' else LABEL[k]) for k in methods],rotation=35,ha='right');ax.set_yticks(range(len(scenarios)),[s.replace('_',' ') for s in scenarios]);ax.set_title(title,fontsize=11)
        threshold=(a.to_numpy().min()+a.to_numpy().max())/2
        for i in range(len(scenarios)):
            for j in range(len(methods)):
                v=a.iloc[i,j];dark=np.dot(im.cmap(im.norm(v))[:3],[.2126,.7152,.0722])<.5
                ax.text(j,i,f'{v:.2f}' if metric=='f1' else f'{v:.4f}',ha='center',va='center',color='white' if dark else '#222222',fontsize=9)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Known-structure simulations: five common data replicates per scenario',fontsize=14)
    fig.supxlabel('Exact 256-point binary-grid audit; estimated effect cutoff = 0.005 probability units.\nLogistic links can create higher-order probability effects; F1 is not native log-odds structure recovery. Five replicates are descriptive.',fontsize=9)
    save(fig,out,'simulation_structure_recovery')
    fidelity=pd.read_csv(source/'shap_fidelity.csv');stability=pd.read_csv(source/'paired_training_stability.csv')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for model,color,marker in [('q2','#2864A0','o'),('unrestricted','#B97718','s')]:
        d=fidelity[fidelity.model==model].groupby('permutations').attribution_rmse.agg(['mean','min','max'])
        axes[0].errorbar(d.index,d['mean'],yerr=[d['mean']-d['min'],d['max']-d['mean']],fmt=marker+'-',color=color,label=LABEL[model],capsize=3)
    axes[0].set_xscale('log');axes[0].set_xticks([1,8,32],['1','8','32']);axes[0].set_yscale('symlog',linthresh=1e-15);axes[0].set_ylim(bottom=0,top=max(fidelity.attribution_rmse)*1.7);axes[0].set_xlabel('Permutation cycles');axes[0].set_ylabel('Attribution RMSE (symlog; linear below 1e-15)');axes[0].set_title('SHAP approximation error');axes[0].legend(frameon=False,fontsize=9)
    for ax,metric,title in zip(axes[1:],['rmse','normalized_rmse'],['Training-resample attribution RMSE','After per-company L1 normalization']):
        for _,g in stability.groupby(['replicate_a','replicate_b']):
            gg=g.set_index('model');ax.plot([0,1],[gg.loc['q2',metric],gg.loc['unrestricted',metric]],color='#BBBBBB',alpha=.6,lw=.7)
        for j,model in enumerate(['q2','unrestricted']):
            y=stability[stability.model==model][metric];ax.scatter(np.full(len(y),j),y,s=15,color=['#2864A0','#B97718'][j],alpha=.65)
            ax.scatter(j,y.median(),marker='_',s=300,color='#222222',zorder=4)
        ax.set_xticks([0,1],['XAI q2','Unrestricted RSF']);ax.set_xlim(-.4,1.4);ax.set_ylabel('Difference between paired refits');ax.set_title(title,fontsize=10);ax.set_ylim(bottom=0)
    for ax in axes:ax.grid(axis='y',color='#DDDDDD',lw=.6)
    fig.suptitle('Separate approximation error from data-resampling instability',fontsize=14)
    fig.supxlabel('Left: 12 fixed company anchors, 16 fixed background rows, 3 explainer seeds; both methods satisfy local additivity.\nRight: exact CIF Shapley values at 32 anchors under 8 common company bootstraps (28 dependent pairs); descriptive, not confidence intervals.',fontsize=9)
    save(fig,out,'shap_fidelity_stability')
    (out/'figure_metadata.json').write_text(json.dumps(dict(source=str(source),uncertainty='seed min-max or dependent pair scatter; no confidence intervals',native_scale_caveat=True,private_company_rows_published=False),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.source,a.out)
