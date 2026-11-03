"""Copy only disclosure-controlled aggregates; plot public and private separately.

No fitting, scoring, thresholds or choices occur in publication. Private tier
rates are used only in the explicitly private plot and never in public prose.
"""
from pathlib import Path
import argparse,json,os,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

PUBLIC=Path(__file__).resolve().parent
LEVELS=['low','medium','high']
NAMES=['低档','中档','高档']


def configure():
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'pdf.fonttype':3,'font.size':10})


def public_plot(out):
    perf=pd.read_csv(out/'TIER_PERFORMANCE.csv')
    aux=pd.read_csv(out/'HIGH_REST_PERFORMANCE.csv')
    fig,axes=plt.subplots(1,2,figsize=(11.8,6.4),sharey=True)
    for ax,h in zip(axes,[12,24]):
        d=aux.query('index_cause==1 and outcome_cause==1 and months==@h')
        for t,offset,color,label in [('high',-.09,'#c44e52','高档（预设合并报告）'),('not_high',.09,'#4c72b0','其余两档合并')]:
            z=d[d.tier==t]
            valid=z.observed_rate.notna()
            ax.scatter(z.loc[valid,'year']+offset,100*z.loc[valid,'observed_rate'],color=color,s=65,label=label,zorder=3)
            for r in z[valid].itertuples():
                ax.annotate(f'{r.observed_rate:.1%}',(r.year+offset,100*r.observed_rate),xytext=(0,9 if t=='high' else -17),textcoords='offset points',ha='center')
        ax.set(title=f'{h} 个月：先最终退市',xticks=[2023,2024,2025],xlim=(2022.6,2025.4),ylim=(0,85))
        ax.grid(axis='y',alpha=.2)
        for y in [2023,2025]:
            label='NA：随访不足' if y==2025 and h==24 else 'S：小格抑制'
            ax.text(y,44,label,ha='center',va='center',color='#666666',fontsize=9)
        rows=[]
        for tier in LEVELS:
            row=[]
            for y in [2023,2024,2025]:
                r=perf.query('index_cause==1 and outcome_cause==1 and months==@h and year==@y and tier==@tier').iloc[0]
                row.append('NA' if not r.supported else ('S' if str(r.publication_status).startswith('SUPPRESSED') else f'{r.observed_rate:.1%}'))
            rows.append(row)
        table=ax.table(cellText=rows,rowLabels=NAMES,colLabels=['2023','2024','2025'],cellLoc='center',bbox=[.03,-.53,.94,.34])
        table.auto_set_font_size(False);table.set_fontsize(9)
        ax.text(.5,-.16,'原低／中／高三档发生率（未合并）',transform=ax.transAxes,ha='center',fontsize=9)
    axes[0].set_ylabel('观察发生率（%）')
    axes[0].legend(loc='upper left',frameon=False,fontsize=9)
    fig.suptitle('袋外阈值分级：公开发生率与小格抑制',fontsize=15,y=.97)
    fig.text(.5,.91,'探索性；只显示可公开的真实汇总，S 和 NA 均不是零发生率',ha='center',fontsize=10,color='#555555')
    fig.subplots_adjust(top=.86,bottom=.37,left=.08,right=.98,wspace=.15)
    fig.text(.06,.025,'三档内出现 1–4 个事件时，三档事件数、率和提升倍数同时抑制。2024 年高／其余合并满足预设发布规则。\n完整三档统计与图保留私有目录；2025 年 24 个月不评价结局。',fontsize=9,color='#555555')
    for ext in ['png','pdf']:fig.savefig(out/f'TIER_INCIDENCE.{ext}',dpi=180)
    plt.close(fig)


def private_plot(analysis):
    d=pd.read_csv(analysis/'TIER_PERFORMANCE_PRIVATE.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,h in zip(axes,[12,24]):
        for tier,color,name in zip(LEVELS,['#4c72b0','#ddaa33','#c44e52'],NAMES):
            z=d.query('index_cause==1 and outcome_cause==1 and months==@h and tier==@tier')
            ax.plot(z.year,100*z.observed_rate,'o-',color=color,label=name)
        ax.set(title=f'{h} 个月',xticks=[2023,2024,2025],xlabel='评价入组年',ylim=(0,100));ax.grid(alpha=.2)
    axes[0].set_ylabel('观察先最终退市发生率（%）');axes[0].legend()
    fig.suptitle('私有：完整三档发生率（含小格，不得公开；无公司标识）')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(analysis/f'TIER_INCIDENCE_PRIVATE.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis-name',default='analysis_final');p.add_argument('--public-output',type=Path,default=PUBLIC);a=p.parse_args()
    if Path(a.analysis_name).name!=a.analysis_name:raise ValueError('Single directory name required')
    run=Path(os.environ.get('RISK_INDEX_RUN',json.loads((PUBLIC/'run_location.json').read_text())['private_run']))
    source=run/a.analysis_name
    target=a.public_output;target.mkdir(parents=True,exist_ok=True)
    for f in (source/'public').glob('*.csv'):
        dest=target/f.name
        if dest.exists():raise FileExistsError(f'Refuse overwrite: {dest}')
        shutil.copy2(f,dest)
    pd.DataFrame(json.loads((run/'REFERENCE_GATE.json').read_text())).to_csv(target/'REFERENCE_GATE.csv',index=False)
    configure();public_plot(target);private_plot(source)
    print('Published suppression-controlled tables/plots only; full tier figure remains private.')


if __name__=='__main__':main()
