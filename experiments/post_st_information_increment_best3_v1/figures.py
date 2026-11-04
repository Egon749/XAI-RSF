"""Publication figures from the disclosed aggregate result tables, no company records."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
COL=['#53697c','#aa552c','#228475','#3e4581']
ARM=['M0 · F','M1 · F + audit','M2 · F + market','M3 · full']
BASE='C039 (17, 503, 22), 96 trees | 12m first termination | exposed cohorts; audit release times incompletely verified.'
def save(fig,tag,note):
    dest=D/'figures';dest.mkdir(exist_ok=True)
    fig.text(.02,.013,BASE+'\n'+note,fontsize=8,va='bottom')
    fig.tight_layout(rect=(0,.12,1,.92))
    for ext in ['pdf','png']:fig.savefig(dest/(tag+'.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
def make():
    a=pd.read_csv(D/'METRICS_ALL.csv').query("cause==1 and months==12 and seed==0 and variant=='fixed25' and year!='pooled'").copy();a.year=a.year.astype(int)
    ci=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv').query("cause==1 and months==12 and metric=='within_year_auc'")
    fig,axes=plt.subplots(1,2,figsize=(10,4));fig.suptitle('Annual discrimination, by available information set')
    for ax,k in zip(axes,['auc','ap']):
        for j,name in enumerate(SETS):
            g=a[a.arm==name].sort_values('year')
            ax.plot(g.year,g[k],marker='o',color=COL[j],label=ARM[j])
        ax.set_xticks(YEARS);ax.set_ylabel(k.upper());ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    save(fig,'01_discrimination','Fixed 25 candidate variables per tree. Annual point estimates after averaging company forecasts.')
    for fam,title,modes in [('audit','Audit information gain',['audit_vs_finance','audit_given_market']),('market','Market information gain',['market_vs_finance','market_given_audit'])]:
        fig,axes=plt.subplots(1,2,figsize=(9.5,4),sharex=True,sharey=True);fig.suptitle(title+' in within-year AUC')
        for ax,e,label in zip(axes,modes,['Over financial baseline','Given the other external group']):
            for i,v in enumerate(['fixed25','proportional']):
                g=ci[(ci.effect==e)&(ci.variant==v)].set_index('year').loc[['2023','2024','2025','pooled']]
                ax.errorbar(g.delta,np.arange(4)+.18*(i-.5),xerr=np.stack([g.delta-g.lower95,g.upper95-g.delta]),fmt='o',ms=4,capsize=3,color=[COL[1],COL[2]][i],label=['Fixed 25','Proportional'][i])
            ax.axvline(0,color='gray',lw=.8);ax.set_yticks(range(4),['2023','2024','2025','Stratified']);ax.set_title(label);ax.set_xlabel('AUC difference')
        axes[0].invert_yaxis();axes[1].legend(fontsize=8)
        save(fig,'02_audit_gain' if fam=='audit' else '03_market_gain','Bars: conditional 95% intervals from 1,000 paired year-stratified company draws. No seed-selection adjustment.')
    fig,axes=plt.subplots(1,2,figsize=(10,4));fig.suptitle('Does the joint gain exceed the sum of the separate gains?')
    z=pd.read_csv(D/'METRICS_ALL.csv').query("year=='pooled' and months==12 and cause==1 and variant=='fixed25' and seed==0").set_index('arm')
    cells=np.array([[z.at['M0','within_year_auc'],z.at['M2','within_year_auc']],[z.at['M1','within_year_auc'],z.at['M3','within_year_auc']]])
    axes[0].imshow(cells,cmap='Blues',vmin=.48,vmax=1,aspect='auto')
    for (i,j),v in np.ndenumerate(cells):axes[0].text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.77 else 'black')
    axes[0].set_xticks([0,1],['No market','Market']);axes[0].set_yticks([0,1],['No audit','Audit']);axes[0].set_title('Within-year AUC, fixed 25')
    for i,v in enumerate(['fixed25','proportional']):
        g=ci[(ci.effect=='complementarity')&(ci.variant==v)].set_index('year').loc[['2023','2024','2025','pooled']]
        axes[1].errorbar(g.delta,np.arange(4)+(i-.5)*.18,xerr=[g.delta-g.lower95,g.upper95-g.delta],fmt='o',capsize=3,ms=4,color=COL[i+1],label=['Fixed 25','Proportional'][i])
    axes[1].set_yticks(range(4),['2023','2024','2025','Stratified']);axes[1].invert_yaxis();axes[1].axvline(0,color='grey',lw=.8);axes[1].legend(fontsize=7);axes[1].set_xlabel('M3 − M2 − M1 + M0 on AUC scale')
    save(fig,'04_factorial_interaction','Right: paired conditional 95% intervals on a performance contrast, not a causal interaction.')
    fig,ax=plt.subplots(figsize=(8,4));fig.suptitle('Captured terminations at the annual top-20% quota')
    for j,n in enumerate(SETS):
        g=a[a.arm==n].sort_values('year');bars=ax.bar(np.arange(3)+(j-1.5)*.19,g.captured,.18,color=COL[j],label=ARM[j])
        ax.bar_label(bars,[f'{int(x)}/{int(y)}' for x,y in zip(g.captured,g.target_events)],fontsize=8,padding=2)
    ax.set_xticks(range(3),YEARS);ax.set_ylim(0,9.4);ax.legend(ncols=2,fontsize=8);ax.set_ylabel('Number captured')
    save(fig,'05_top20','Each model reviews 8, 9 and 11 companies respectively, ranked within the entry year.')
    fig,ax=plt.subplots(figsize=(8,4));fig.suptitle('Mean predicted risk and observed termination incidence')
    for j,n in enumerate(SETS):
        g=a[a.arm==n].sort_values('year');ax.plot(g.year,100*g.risk_mean,marker=['o','x','+','s'][j],color=COL[j],label=ARM[j])
    ax.plot(g.year,100*g.observed,'kD--',label='Observed');ax.set_xticks(YEARS);ax.set_ylabel('Termination (%)');ax.legend(fontsize=8,ncols=2)
    save(fig,'06_probability_level','No calibrator fitted. Means use all evaluation firms with supported 12-month outcomes.')
if __name__=='__main__':make()
