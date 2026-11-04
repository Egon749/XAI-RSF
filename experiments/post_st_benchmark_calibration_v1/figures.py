import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *
import pandas as pd
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
COL=['#165c73','#c26332','#546371']
def save(fig,n):
    d=D/'figures';d.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(d/(n+'.'+ext),dpi=190,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def main():
    a=pd.read_csv(D/'PAIRED_DIFFERENCES.csv');names={'AJ':'Aalen–Johansen','KNN_AJ':'Nearest-neighbor AJ','CSC':'Cause-specific Cox','FG':'Fine–Gray','unrestricted/selected_seeds':'Same-backend, q unrestricted','RF/selected_seeds':'Standard RSF'}
    fig,ax=plt.subplots(figsize=(8.6,4.5))
    for i,(ref,label) in enumerate([('C039','Selected C039'),('median120','Median of 120 triples')]):
        s=a.query("scope=='pooled' and months==12 and cause==1 and metric=='within_year_auc' and reference==@ref").set_index('comparator').loc[list(names)]
        y=np.arange(len(s))+(i-.5)*.19
        ax.errorbar(s.difference,y,xerr=np.vstack([s.difference-s.lower95,s.upper95-s.difference]),fmt=['o','s'][i],color=COL[i],capsize=2,label=label)
    ax.axvline(0,color='#788692',lw=1);ax.set_yticks(range(len(names)),names.values());ax.set_xlabel('XAI-RSF minus comparator, within-year AUC');ax.legend(loc='upper right',fontsize=9);ax.set_title('12-month delisting discrimination across 2023–2025',loc='left',fontweight='bold')
    fig.text(.02,-.055,'134 firms, 25 events. Conditional paired 95% intervals from 1,000 shared company draws. Forest comparators use 17/503/22.',fontsize=8);save(fig,'01_benchmark_differences')
    cal=pd.read_csv(D/'CALIBRATION.csv');fig,axes=plt.subplots(1,2,figsize=(10.4,4.5),sharey=True)
    for ax,(c,lab) in zip(axes,[(1,'Final delisting'),(2,'Recovery')]):
        s=cal[(cal.cause==c)&(cal.scope!='pooled')&cal.supported].copy().sort_values(['months','scope']);pos=np.arange(len(s))
        for j,rr in enumerate(s.itertuples()):ax.plot([pos[j]]*2,[rr.mean_predicted*100,rr.observed*100],c='#a7b5bb')
        ax.scatter(pos,s.mean_predicted*100,color=COL[0],label='Mean C039 prediction',s=45);ax.scatter(pos,s.observed*100,color=COL[1],marker='D',label='Observed',s=35)
        ax.set_xticks(pos,[f'{y}\n{m}m' for y,m in zip(s.scope,s.months)]);ax.set_title(lab,loc='left',fontweight='bold');ax.set_ylim(-2,64)
    axes[0].set_ylabel('Cumulative incidence (%)');axes[1].legend(loc='upper left',fontsize=8);fig.suptitle('Probability levels at the supported calendar horizons',x=.09,ha='left',weight='bold')
    fig.text(.03,-.015,'2025 at 24 months is withheld for incomplete follow-up. Ranking and risk-index scaling do not recalibrate these probabilities.',fontsize=8);save(fig,'02_probability_levels')
    tiers=pd.read_csv(D/'RISK_INDEX.csv');fig,ax=plt.subplots(figsize=(7.2,3.6));s=tiers.query("months==12 and index_cause==1 and outcome_cause==1")
    left=np.zeros(3)
    for name,col in [('low','#cddce1'),('medium','#6f9cab'),('high',COL[0])]:
        z=s[s.tier==name].sort_values('year').share.to_numpy()*100;ax.barh(range(3),z,left=left,label=name.title(),color=col);left+=z
    ax.set_yticks(range(3),['2023 (36)','2024 (45)','2025 (53)']);ax.set_xlim(0,100);ax.set_xlabel('Share of the entry cohort (%)');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.21),frameon=False);ax.set_title('Where the OOB-defined risk tiers place each cohort',loc='left',weight='bold')
    fig.subplots_adjust(bottom=.31);fig.text(.03,.022,'12-month delisting index; cutoffs estimated in each training snapshot. The high tier has no fixed test quota.',fontsize=8);save(fig,'03_oob_tiers')
if __name__=='__main__':main()
