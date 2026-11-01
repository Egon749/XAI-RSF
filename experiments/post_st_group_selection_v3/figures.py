"""Aggregate process and prediction plots; no company data or fitted thresholds."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import PUBLIC,RUN


def main():
    tasks=pd.read_csv(PUBLIC/'PIPELINE_RUNS.csv');boot=tasks[tasks.repeat>=0]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    years=[2023,2024,2025];search=[int(boot[boot.year==y].supervised_search_executed.sum()) for y in years]
    selected=[int((boot[boot.year==y].features<83).sum()) for y in years]
    axes[0].bar([str(y) for y in years],[50-n for n in search],label='Insufficient discovery support',color='#aab5be')
    axes[0].bar([str(y) for y in years],search,bottom=[50-n for n in search],label='Actual candidate refitting/search',color='#2980b9')
    for i,n in enumerate(search):axes[0].text(i,51,f'{n}/50 searched',ha='center',fontsize=10)
    axes[0].set_ylim(0,60);axes[0].set_ylabel('Company resampling pipelines');axes[0].legend(loc='upper left',fontsize=8)
    axes[1].bar([str(y) for y in years],selected,color='#208467')
    for i,n in enumerate(selected):axes[1].text(i,n+.5,str(n),ha='center')
    axes[1].set_ylim(0,50);axes[1].set_ylabel('Pipelines selecting fewer than 83 inputs')
    fig.suptitle('Complete selection process: 50 company resamples per cutoff')
    fig.savefig(PUBLIC/'selection_process.png',dpi=180);plt.close(fig)
    data=pd.read_csv(PUBLIC/'MODEL_COMPARISON.csv');data=data[(data.seed=='ensemble')&(data.cause==1)]
    roles=['full83','legacy74','selected_pipeline'];labels=['83 inputs','74 inputs','Guarded selection'];colors=['#4c78a8','#f28e2b','#23967f']
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for i,h in enumerate([12,24]):
        for j,metric in enumerate(['auc','ap','brier']):
            ax=axes[i,j]
            for n,(role,label,color) in enumerate(zip(roles,labels,colors)):
                rows=data[(data.model==role)&(data.horizon==h)].copy();values=[]
                for y in years:
                    z=rows[rows.year.astype(str)==str(y)];values.append(float(z.iloc[0][metric]))
                ax.bar(np.arange(3)+(n-1)*.23,values,width=.23,color=color,label=label)
            ax.set_xlim(-.6,2.6);ax.set_xticks(range(3),map(str,years));ax.set_title(f'{h} months: {metric.upper()}');ax.set_ylim(0,1 if metric!='brier' else .45)
            if h==24:ax.text(2,.1,'NA: immature',ha='center',fontsize=8)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8);fig.suptitle('Exploratory first-termination prediction; three-seed ensemble')
    fig.savefig(PUBLIC/'annual_prediction_comparison.png',dpi=180);plt.close(fig)
    for name in ['selection_process.png','annual_prediction_comparison.png']:shutil.copy2(PUBLIC/name,RUN/'reports'/name)

if __name__=='__main__':main()
