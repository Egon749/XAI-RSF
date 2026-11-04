"""Figures generated solely from auditable tables or specified private case files."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common import *
COL=['#276787','#d8783a','#4b9270']
GNAME=['Profitability','Solvency','Operations','Growth','Cash flow','Market','Audit','Controls']
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def primary(name):
    return pd.read_csv(D/(name+'.csv')).query('months==12 and cause==1')
def finish(fig,name,caption,out):
    fig.text(.015,.015,caption,va='bottom',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.94))
    for ext in ['pdf','png']:fig.savefig(out/(name+'.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
def plot():
    out=D/'figures';out.mkdir(exist_ok=True)
    m=primary('GROUP_MODEL_DISTRIBUTIONS');z=primary('GROUP_MAIN_CONTRIBUTIONS').query("level=='C8' and shared==0")
    four=primary('FOUR_CAPABILITIES_STABILITY').query('shared==0')
    base='Shrunk 96-tree triples | 12-calendar-month termination CIF | actual evaluation firms, no training joint-support filter; training marginal-product centering.'
    fig,ax=plt.subplots(figsize=(8,4));fig.suptitle('Four financial capabilities: magnitude across 120 seed triples')
    for i,y in enumerate(YEARS):
        d=m[(m.year==y)&(m.level=='C4')&m.model.isin(COMB)]
        data=[100*d.loc[d.group==j,'mean_abs'] for j in LABELS[:4]]
        ax.boxplot(data,positions=np.arange(4)+(i-1)*.22,widths=.18,whis=(5,95),showfliers=False,patch_artist=True,boxprops=dict(facecolor=COL[i],alpha=.75),medianprops=dict(color='black'))
    ax.set_xticks(range(4),GNAME[:4]);ax.set_ylabel('Mean absolute attribution (CIF pp)')
    ax.legend(handles=[Line2D([],[],color=c,lw=5,label=str(y)) for c,y in zip(COL,YEARS)])
    finish(fig,'01_financial_magnitude',base+'\nBoxes: interquartile range and median; whiskers: P5–P95 of the finite seed set.',out)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True);fig.suptitle('Group contributions and agreement on direction')
    for i,y in enumerate(YEARS):
        a=z[z.year==y].set_index('group').loc[LABELS];idx=np.arange(8)+.16*(i-1);mid=a.mean_abs_median.to_numpy()
        axes[0].errorbar(100*mid,idx,xerr=100*np.stack([mid-a.amp_p05,a.amp_p95-mid]),fmt='o',ms=3,capsize=2,color=COL[i])
        axes[0].scatter(100*a.ref96_mean_abs,idx,marker='x',color=COL[i],s=18)
        axes[1].scatter(a.sign_agreement,idx,color=COL[i],label=str(y))
    axes[0].set_yticks(range(8),GNAME);axes[0].invert_yaxis();axes[0].set_xlabel('Mean |C| (pp); × = original 96 trees');axes[1].set_xlabel('Sign agreement when both material');axes[1].legend()
    finish(fig,'02_eight_groups',base+'\nBars: P5–P95 across 120 triples. Direction averages over firms, disjoint triples, |C| ≥ 0.1 pp for both.',out)
    inter=primary('GROUP_INTERACTIONS').query('shared==0')
    fig,axes=plt.subplots(2,3,figsize=(11,7));fig.suptitle('Between-group interactions: magnitude and conditional agreement')
    for i,(metric,title) in enumerate([('mean_abs_median','Mean |I| (pp)'),('sign_agreement','Sign agreement')]):
        vmax=inter[metric].max()*100 if i==0 else 1.
        for j,y in enumerate(YEARS):
            arr=np.full((8,8),np.nan)
            for pair,row in inter[inter.year==y].set_index('group').iterrows():
                a,b=[int(g[1:])-1 for g in pair.split('×')];arr[a,b]=arr[b,a]=row[metric]*(100 if i==0 else 1)
            ax=axes[i,j];im=ax.imshow(np.ma.masked_invalid(arr),vmin=0,vmax=vmax,cmap='viridis');ax.set_xticks(range(8),LABELS,fontsize=7);ax.set_yticks(range(8),LABELS,fontsize=7);ax.set_title(f'{y}   {title}');fig.colorbar(im,ax=ax,fraction=.046)
    finish(fig,'03_interaction_matrix',base+'\nUpper panels: median model-wide magnitude. Lower: disjoint comparisons with both contributions material; undefined cells blank. No uncertainty bands.\nG1–G8: profitability, solvency, operations, growth, cash flow, market, audit, controls.',out)
    comp=primary('GROUP_VS_FEATURE_COMPARISON').query('shared==0')
    fig,axes=plt.subplots(2,3,figsize=(11,6));fig.suptitle('Economic mapping versus equally sized random maps and raw variables')
    for i,lev in enumerate(['C8','C4']):
        for j,(met,text) in enumerate([('relative_l1','Normalized L1 (lower)'),('rank_rho','Absolute rank ρ (higher)'),('top_jaccard','Top-set Jaccard')]):
            ax=axes[i,j];a=comp[(comp.level==lev)&(comp.metric==met)].sort_values('year')
            mid=a.random_median.to_numpy();ax.errorbar(a.year-.08,mid,yerr=np.stack([mid-a.random_p05,a.random_p95-mid]),fmt='o',color='#7f8d9d',capsize=4,label='Random maps');ax.scatter(a.year,a.economic,color=COL[0],label='Economic');ax.scatter(a.year+.08,a.variable_full74 if i==0 else a.financial_variables35,color=COL[1],marker='x',label='Variables')
            ax.set_xticks(YEARS);ax.set_title(('Eight groups' if i==0 else 'Four capabilities')+' | '+text)
    axes[0,0].legend(fontsize=7)
    finish(fig,'04_random_controls',base+'\nRandom P5–P95 across 200 fixed maps; dots are mean company-level stability on 2,100 disjoint pairs. Top-k differs (8 groups: 3; four: 1; variables: 5).',out)
    profiles=primary('LOCAL_CASE_BANDS').query('year==2024')
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True);fig.suptitle('2024 local explanations in preselected low/high variability cohorts')
    for ax,band in zip(axes,['low','high']):
        a=profiles[profiles.band==band].set_index('group').loc[LABELS];mid=a.signed_median.to_numpy()
        ax.errorbar(100*mid,np.arange(8),xerr=100*np.stack([mid-a.p05,a.p95-mid]),fmt='o',color=COL[0],capsize=3);ax.axvline(0,color='grey',lw=.7)
        ax.set_yticks(range(8),GNAME);ax.set_title(f'{band.capitalize()} variability, n={int(a.companies.iloc[0])}');ax.set_xlabel('Signed C (CIF pp)')
    axes[0].invert_yaxis();finish(fig,'05_local_group_profiles',base+'\nP5–P95 across triples of the within-cohort signed mean. Real individual charts are private.',out)
    fig,axes=plt.subplots(1,3,figsize=(11,4));fig.suptitle('Financial capability attribution across annual entry cohorts')
    for j,g in enumerate(LABELS[:4]):
        a=four[four.group==g].sort_values('year')
        for ax,metric,factor,title in zip(axes,['median_rank','mean_abs_median','signed_mean_median'],[1,100,100],['Rank among four','Mean absolute C (pp)','Mean signed C (pp)']):
            ax.plot(a.year,a[metric]*factor,marker='o',label=GNAME[j]);ax.set_xticks(YEARS);ax.set_title(title)
    axes[0].invert_yaxis();axes[1].legend(fontsize=7);axes[2].axhline(0,color='grey',lw=.7)
    finish(fig,'06_across_years',base+'\nPoints: medians across 120 triples; lines connect distinct annual cohorts. No interval bands or same-firm longitudinal comparison.',out)
    dest=run()/'private_case_figures';dest.mkdir(exist_ok=True)
    for rec in read(run()/'analysis/published.json'):
        if rec['args'][2]!='0':continue
        y=int(rec['args'][1]);p=Path(rec['path'])
        for months in (12,24):
            for cause in (1,2):
                for q in (.1,.9):
                    a=arrays(p/f'case_{months}_{cause}_{q}_PRIVATE.npz');lo,mid,hi=100*np.quantile(a['C'],[.05,.5,.95],axis=0)
                    fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(mid,np.arange(8),xerr=np.stack([mid-lo,hi-mid]),fmt='o',capsize=3,color=COL[0])
                    ax.scatter(100*a['reference'],np.arange(8),marker='x',color=COL[1]);ax.axvline(0,color='gray',lw=.7);ax.set_yticks(range(8),GNAME);ax.set_xlabel('Signed contribution (CIF pp)')
                    fig.suptitle(f"Private case | {y}, {months}m {'termination' if cause==1 else 'recovery'} | prior L1 percentile {q*100:g}")
                    finish(fig,f'{y}_{months}_{cause}_{q}_PRIVATE','Actual evaluation firm; P5–P95 over 120 shrunk triples, × original 96; training marginal-product reference; no joint-support filter. Descriptive range.',dest)
if __name__=='__main__':plot()
