"""Draw only admissible point estimates. Missing arms are explicitly labelled."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
BLUE='#224d70';TEAL='#008579';GRAY='#88949a';RED='#b35435'
def save(fig,name,caption):
    out=D/'figures';out.mkdir(exist_ok=True);fig.text(.02,.012,caption,fontsize=8,va='bottom')
    fig.tight_layout(rect=(0,.10,1,.94))
    for ext in ['pdf','png']:fig.savefig(out/(name+'.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
def make():
    met=pd.read_csv(D/'PREDICTIONS_AND_METRICS.csv')
    q=met.query("seed==0 and cause==1 and months==12 and variant=='fixed25' and arm in ['M0','M2'] and year!='pooled'").copy();q.year=q.year.astype(int)
    boot=pd.read_csv(D/'PAIRED_DIFFERENCES_95.csv')
    baseline='12-calendar-month final termination | 96 trees per year, matched companies, post-exposure. M1/M3 blocked by audit disclosure timing.'
    fig,ax=plt.subplots(1,2,figsize=(9,3.7));fig.suptitle('Discrimination by information set and entry cohort')
    for a,k in zip(ax,['auc','ap']):
        for name,col in [('M0',BLUE),('M2',TEAL)]:
            z=q[q.arm==name].sort_values('year');a.plot(z.year,z[k],marker='o',color=col,label=name)
        a.set_xticks(YEARS);a.set_ylabel(k.upper());a.legend();a.grid(alpha=.16)
    save(fig,'01_annual_discrimination',baseline)
    fig,ax=plt.subplots(figsize=(8,3.5));fig.suptitle('Incremental within-year AUC, 12m termination')
    z=boot.query("months==12 and cause==1 and variant=='fixed25' and metric=='within_year_auc' and year=='pooled'").set_index('effect')
    targets=list(['audit_vs_finance','market_vs_finance','audit_given_market','market_given_audit','complementarity','market_numeric_vs_finance','coverage_only_vs_finance'])
    for i,k in enumerate(targets):
        a=z.loc[k]
        if pd.isna(a.delta):ax.text(.005,i,'NA · audit timing',color=GRAY,va='center')
        else:ax.errorbar(a.delta,i,xerr=[[a.delta-a.lower95],[a.upper95-a.delta]],fmt='o',capsize=4,color=TEAL)
    ax.set_yticks(range(len(targets)),['Audit | F','Market | F','Audit | F + market','Market | F + audit','Performance interaction','Numerical market | F','Coverage only | F']);ax.axvline(0,color=GRAY,lw=.9);ax.set_ylim(len(targets)-.55,-.55)
    save(fig,'02_paired_increments','Fixed predictions. Conditional 95% percentile intervals from 1,000 common company draws stratified by year. Higher = better.')
    fig,ax=plt.subplots(figsize=(6,3.7));fig.suptitle('Financial baseline with audit and market access')
    z=met.query("seed==0 and cause==1 and months==12 and variant=='fixed25' and year=='pooled'").set_index('arm')
    vals=np.array([[z.at['M0','within_year_auc'],z.at['M2','within_year_auc']],[np.nan,np.nan]])
    cm=plt.get_cmap('Blues').copy();cm.set_bad('#eeeeee');ax.imshow(vals,vmin=.45,vmax=.85,cmap=cm,aspect='auto')
    for i in range(2):
        for j in range(2):
            label=[['M0','M2'],['M1','M3']][i][j];ax.text(j,i,label+'\n'+(f'{vals[i,j]:.3f}' if i==0 else 'NA · source gate'),ha='center',va='center',fontsize=13)
    ax.set_xticks([0,1],['No market','Market available']);ax.set_yticks([0,1],['No audit','Audit available'])
    save(fig,'03_factorial','Within-year AUC, 12m termination. Two cells unavailable; performance interaction cannot be estimated.')
    fig,ax=plt.subplots(figsize=(7,3.6));fig.suptitle('Events captured by the annual top 20%')
    for j,a in enumerate(['M0','M2']):
        z=q[q.arm==a].sort_values('year');pos=np.arange(3)+(j-.5)*.32
        bars=ax.bar(pos,z.captured,.32,color=[BLUE,TEAL][j],label=a);ax.bar_label(bars,[f'{int(v)}/{int(e)}' for v,e in zip(z.captured,z.target_events)],padding=2)
    ax.set_xticks(range(3),YEARS);ax.set_ylim(0,7);ax.set_ylabel('Captured terminations');ax.legend()
    save(fig,'04_top20',baseline)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8));fig.suptitle('Absolute risk and observed incidence')
    zz=q[q.arm=='M0'].sort_values('year');axes[0].plot(YEARS,100*zz.observed,'ko-',label='Observed')
    for a,col in [('M0',BLUE),('M2',TEAL)]:
        z=q[q.arm==a].sort_values('year');axes[0].plot(YEARS,100*z.risk_mean,'o-',color=col,label=a)
        cal=pd.read_csv(D/'CALIBRATION_DIAGNOSTICS.csv').query("year==2024 and months==12 and cause==1 and arm==@a and variant=='fixed25'").sort_values('risk_quartile_high_first')
        axes[1].plot(100*cal.prediction,100*cal.observed,'o-',color=col,label=a)
    axes[0].set_xticks(YEARS);axes[0].set_ylabel('Termination (%)');axes[0].legend();axes[1].plot([0,65],[0,65],':',color=GRAY);axes[1].set(xlabel='Quartile mean risk (%)',ylabel='Quartile incidence (%)',title='2024 cohort');axes[1].legend()
    save(fig,'05_calibration',baseline)
    cx=pd.read_csv(D/'INTERPRETABILITY_COMPLEXITY.csv').query("seed==0 and version=='shrunk' and variant=='fixed25' and arm in ['M0','M2']")
    d=q.merge(cx,on=['year','arm','seed','variant']);fig,ax=plt.subplots(figsize=(7,3.7));fig.suptitle('Predictive ranking and explanation size')
    for y,g in d.groupby('year'):
        g=g.sort_values('arm');ax.plot(g.interaction_functions,g.auc,'o-',label=str(y))
        for row in g.itertuples():ax.annotate(row.arm,(row.interaction_functions,row.auc),xytext=(3,4),textcoords='offset points')
    ax.margins(x=.09,y=.17);ax.legend();ax.set_xlabel('Nonzero bivariate functions');ax.set_ylabel('Annual AUC')
    save(fig,'06_complexity',baseline)
if __name__=='__main__':make()
