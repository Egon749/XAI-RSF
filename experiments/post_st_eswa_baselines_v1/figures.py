"""Paper figures generated only from released aggregate results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .common import *
models=['XAI/C039','XAI/anchor','PROBoost/IPCW','Stacking/IPCW'];labels=['XAI-RSF C039*','XAI-RSF anchor','PROBoost-H','Chen stacking-H'];colors=['#105e70','#72949b','#c3642d','#795a87'];marks=['o','s','D','^']
def save(fig,name,caption):
    fig.text(.055,.015,caption,fontsize=8,color='#415469');folder=D/'figures';folder.mkdir(exist_ok=True)
    for ext in ('svg','png'):
        target=folder/(name+'.'+ext);fig.savefig(target,dpi=200,bbox_inches='tight',facecolor='white',metadata={'Date':None} if ext=='svg' else {})
        if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)
def main():
    a=pd.read_csv(D/'METRICS_BY_YEAR.csv',dtype={'scope':str});p=pd.read_csv(D/'POOLED_METRICS.csv');df=pd.concat([a,p]);cost=pd.read_csv(D/'RUNTIME_AND_COMPLEXITY.csv')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'eswa-baselines-v1','axes.spines.top':False,'axes.spines.right':False})
    def panel(ax,h,metric,scopes,top=False):
        for j,(model,label,col,marker) in enumerate(zip(models,labels,colors,marks)):
            g=df[(df.months==h)&(df.model==model)].set_index('scope');vals=[g.at[k,metric] for k in scopes]
            x=np.arange(len(scopes))+(j-1.5)*.13;ax.scatter(x,vals,color=col,marker=marker,label=label,s=48)
        n=['2023\n4 events','2024\n16 events','2025\n5 events','Pooled\n25 events'] if h==12 else ['2023\n12 events','2024\n20 events','Pooled\n32 events']
        ax.set_xticks(np.arange(len(scopes)),[dict(zip((['2023','2024','2025','pooled'] if h==12 else ['2023','2024','pooled']),n))[k] for k in scopes]);ax.set_xlim(-.5,len(scopes)-.5);ax.grid(axis='y',alpha=.2)
        if top:
            ax.set_ylim(-.7,16);ax.set_yticks([0,3,6,9,12,15])
        else:ax.set_ylim(0,1.05)
    fig,ax=plt.subplots(figsize=(7.4,4.5));panel(ax,12,'auc',['2023','2024','2025']);ax.axhline(.5,lw=.8,ls='--',c='#7a8189');ax.set_ylabel('Annual ROC-AUC');ax.set_title('12-month delisting discrimination by entry cohort',loc='left',weight='bold');ax.legend(ncol=2,fontsize=8,loc='lower left');fig.subplots_adjust(bottom=.2);save(fig,'01_annual_auc','*C039 was selected on these exposed cohorts. Classifiers use training IPCW. The anchor was fixed separately.')
    for metric,n,title,unit in [('ap','02_average_precision','Average precision by horizon and entry year','Average precision'),('top20_captured','03_top20_capture','Delisting events captured in annual top-quintile lists','Events captured')]:
        fig,axs=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
        for ax,h,sco in [(axs[0],12,['2023','2024','2025','pooled']),(axs[1],24,['2023','2024','pooled'])]:
            panel(ax,h,metric,sco,metric=='top20_captured');ax.set_title(f'{h} months',loc='left')
        axs[0].set_ylabel(unit);axs[1].legend(loc='upper right' if metric=='ap' else 'upper left',ncol=2,fontsize=7);fig.suptitle(title,weight='bold');fig.subplots_adjust(bottom=.20);save(fig,n,'Separate classifiers per horizon. Pooled AP uses concatenated scores; pooled capture retains the original annual quotas.')
    fig,ax=plt.subplots(figsize=(7.4,4.5));panel(ax,24,'within_year_auc',['2023','2024','pooled']);ax.legend(ncol=2,loc='lower right',fontsize=8);ax.set_ylabel('Within-year ROC-AUC');ax.set_title('24-month delisting ranking',loc='left',weight='bold');fig.subplots_adjust(bottom=.2);save(fig,'04_secondary_auc','81 firms, 32 delisting events; pooled AUC here includes only within-year pairs. 2025 H24 is not scored.')
    fig,ax=plt.subplots(figsize=(8,4.7));offset=[(7,5),(7,-10),(6,6),(-95,7)]
    for j,m in enumerate(models):
        if j<2:z=cost[cost.model==m].historical_fit_seconds.sum()
        else:z=cost.query("model==@arm and mode=='IPCW' and months=='12'",local_dict={'arm':m.split('/')[0]}).end_to_end_seconds.sum()
        u=p[(p.model==m)&(p.months==12)].within_year_auc.iloc[0];ax.scatter(z,u,color=colors[j],marker=marks[j],s=65);ax.annotate(labels[j],(z,u),xytext=offset[j],textcoords='offset points',fontsize=9)
    ax.set_xscale('log');ax.set_xticks([2,5,10,20,50],['2','5','10','20','50']);ax.minorticks_off();ax.margins(x=.27,y=.22);ax.set_ylabel('12-month within-year AUC');ax.set_xlabel('Summed seconds across three cohorts (log scale)');ax.set_title('Ranking performance and observed computation',loc='left',weight='bold');fig.subplots_adjust(bottom=.19);save(fig,'05_computational_cost','Classifiers include search, refits and scoring. XAI uses archived member fit times only. Machines and timing boundaries differ.')
    fig,ax=plt.subplots(figsize=(9,3.8));ax.axis('off');vals=[]
    for j,m in enumerate(models):
        u=p[(p.model==m)&(p.months==12)].iloc[0];vals.append([labels[j],f'{u.within_year_auc:.3f}',f'{u.ap:.3f}','Yes' if j<2 else 'No','Exact' if j<2 else 'No native table','≤2' if j<2 else 'No guarantee'])
    tab=ax.table(cellText=vals,colLabels=['Predictor','Within-year AUC','Pooled AP','Joint CIF','Global 1D/2D CIF','Interaction order¹'],cellLoc='center',loc='center',colWidths=[.18,.17,.13,.12,.21,.17]);tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.65)
    for (i,j),cell in tab.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor('#205665' if i==0 else ['#e7eff0','#f3f6f6'][i%2])
        if not i:cell.get_text().set_color('white')
    ax.set_title('What each predictor explains at the reported output scale',loc='left',weight='bold');save(fig,'06_structure','¹At the final CIF/score scale. TreeSHAP can exactly allocate a compatible tree margin without imposing a global two-variable representation.')
if __name__=='__main__':main()
