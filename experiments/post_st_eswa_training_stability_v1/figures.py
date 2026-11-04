import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .run import D
def main():
    tab=pd.read_csv(D/'SIDE_BY_SIDE.csv',dtype={'scope':str});out=D/'figures';out.mkdir(exist_ok=True)
    for h,scopes in [(12,['2023','2024','2025','pooled']),(24,['2023','2024','pooled'])]:
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),sharey=True)
        for ax,metric,title in zip(axes,['test_within_year_auc','test_ap'],['Within-year AUC','Average precision']):
            for j,(name,col) in enumerate([('PROBoost','#b35f32'),('Stacking','#655485'),('XAI-RSF/120','#125c70')]):
                z=tab[(tab.months==h)&(tab.metric==metric)&(tab.model==name)].set_index('scope').loc[scopes];y=np.arange(len(z))+(j-1)*.18
                ax.hlines(y,z.minimum,z.maximum,colors=col,linewidth=.8,alpha=.35)
                ax.hlines(y,z.p05,z.p95,colors=col,linewidth=3)
                ax.scatter(z['median'],y,color=col,s=35,label=name,zorder=3)
                ax.scatter(z.original,y,marker='|',s=90,c='#202830',zorder=4)
            ax.set_xlim(-.03,1.03);ax.set_yticks(range(len(scopes)),[s if s!='pooled' else 'Pooled' for s in scopes]);ax.set_title(title,loc='left',weight='bold');ax.grid(axis='x',alpha=.22);ax.spines[['top','right']].set_visible(False)
        axes[0].invert_yaxis();hnd,lbl=axes[0].get_legend_handles_labels();fig.legend(hnd,lbl,ncol=3,loc='lower center',bbox_to_anchor=(.5,.075),frameon=False)
        fig.suptitle(f'{h}-month delisting: original scores and perturbation distributions',weight='bold');fig.subplots_adjust(bottom=.23,top=.81)
        fig.text(.03,.04,'Dot: median; thick: P05–P95; thin: range; black tick: original fit (C039 for XAI-RSF).',fontsize=8)
        fig.text(.03,.006,'ESWA: 50 historical-company resamples. XAI-RSF: 120 overlapping seed triples at fixed training data. Descriptive, not confidence intervals.',fontsize=8)
        fig.savefig(out/f'distributions_{h}m.png',dpi=180,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
