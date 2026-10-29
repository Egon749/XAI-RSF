"""Export aggregate-only scientific figures for the repository report."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def figures(root,output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    cohorts=pd.read_csv(root/'reports/cohort_aj.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharey=True)
    for row,year in enumerate([2023,2024]):
        for col,h in enumerate([12,24]):
            ax=axes[row,col];d=cohorts[(cohorts.cutoff_year==year)&(cohorts.horizon_months==h)&cohorts.cohort.astype(str).isin(['2018','2019','2020','2021','2022','2023'])].copy()
            for i,r in enumerate(d.itertuples()):
                if r.horizon_supported:
                    ax.errorbar(i,r.aj_termination,yerr=[[max(0,r.aj_termination-r.lower_95)],[max(0,r.upper_95-r.aj_termination)]],fmt='o',capsize=4,color='#205ea8')
                    ax.annotate(f'{r.aj_termination:.1%}',(i,r.aj_termination),xytext=(5,7),textcoords='offset points',fontsize=9)
                else:ax.text(i,.024,'NA\n随访不足',ha='center',va='center',color='#8e4d2f',fontsize=9)
            ax.set_xticks(range(len(d)),d.cohort.astype(str));ax.set_ylim(0,.36);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='y',alpha=.2)
            ax.set_title(f'信息截止 {year}-01-01 之前｜{h} 个月');ax.set_xlabel('入组年份')
            if col==0:ax.set_ylabel('训练队列退市 AJ')
    fig.suptitle('预测前的近期队列已有风险上升，但部分时点尚无随访支持',fontsize=14,y=.99)
    fig.text(.08,.015,'来源：冻结训练元数据；展示 2018 年起队列，完整年份见 CSV。点为 AJ，线为公司 bootstrap 95% 区间（500 次）。\nNA 不等于零；零事件区间退化不能证明真实风险为零。入组年份差异不是政策因果效应。',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(output/'training_cohort_aj.png');fig.savefig(output/'training_cohort_aj.pdf');plt.close(fig)

    d=pd.read_csv(root/'summary/termination_ensemble_summary.csv')
    arms=['baseline','time_w2','time_w4','leaf20'];labels=['原深度4＋收缩','近期权重2','近期权重4','min_leaf20']
    colors=['#245ca6','#4b9c91','#c36a3f','#797079']
    fig,axes=plt.subplots(2,2,figsize=(11,7.2))
    for col,year in enumerate(['2023','2024']):
        dd=d[(d.scope.astype(str)==year)&(d.horizon_months==24)].set_index('arm').loc[arms]
        ax=axes[0,col];bars=ax.bar(np.arange(4),dd.risk_mean,color=colors,width=.65)
        actual=float(dd.ipcw_observed_incidence.iloc[0]);ax.axhline(actual,color='#333333',ls='--',label=f'观察发生率 {actual:.1%}')
        ax.bar_label(bars,labels=[f'{v:.1%}' for v in dd.risk_mean],padding=4);ax.set_ylim(0,.52);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,loc='upper left');ax.set_title(f'{year} 年入组｜24 个月');ax.set_ylabel('退市概率');ax.set_xticks(range(4),labels,rotation=10,ha='right')
        ax=axes[1,col]
        ax.plot(range(4),dd.auc,'o-',label='AUC',color='#245ca6');ax.plot(range(4),dd.ap,'s--',label='AP',color='#c36a3f')
        for i,(auc,ap) in enumerate(zip(dd.auc,dd.ap)):
            ax.annotate(f'{auc:.3f}',(i,auc),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
            ax.annotate(f'{ap:.3f}',(i,ap),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xlim(-.3,3.3);ax.set_ylim(0,1);ax.set_xticks(range(4),labels,rotation=10,ha='right');ax.set_ylabel('排序指标');ax.legend(frameon=False,loc='lower left');ax.grid(axis='y',alpha=.2)
    fig.suptitle('时间加权缓解 24 个月风险低估，同时削弱排序',fontsize=14,y=.99)
    fig.text(.08,.015,'来源：36/45 家测试公司，3 个种子的平均预测（每个森林 32 棵树）。2023/2024 已暴露，仅作描述性比较。\n图示全部固定候选；历史选择为 2023 权重1、2024 权重4。12 个月结果和逐种子结果见配套 CSV。',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(output/'prediction_tradeoff.png');fig.savefig(output/'prediction_tradeoff.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();figures(a.run,a.output)
