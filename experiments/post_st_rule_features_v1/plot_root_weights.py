"""Plot only aggregate score diagnostics, with no raw financial cutpoints."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(reports,out):
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,
                         'font.size':11,'svg.fonttype':'none'})
    a=pd.read_csv(reports/'price_same_candidate_scores.csv')
    fig,ax=plt.subplots(figsize=(8.1,4.8),layout='constrained')
    for key,label,color,marker in [('combined','(1,1)：两个结局','#4477AA','o'),
                                  ('termination_only','(1,0)：只看退市','#CC6677','s'),
                                  ('recovery_only','(0,1)：只看恢复','#228833','^')]:
        ax.plot(range(1,len(a)+1),a[key],label=label,color=color,marker=marker,lw=2)
    ax.set(xticks=range(1,len(a)+1),xlabel='同一组冻结股价切点：从低到高编号（不披露真实值）',
           ylabel='根节点分裂分数',ylim=(0,3.7),title='反号抵消确实存在，但组合分数并非总低于退市分数')
    ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',frameon=False)
    ax.annotate('只看退市的最佳切点',xy=(3,a.termination_only.iloc[2]),xytext=(2.8,3.45),
                ha='center',fontsize=10,arrowprops={'arrowstyle':'->','color':'#CC6677'})
    fig.savefig(out/'price_split_scores.png',dpi=180)
    fig.savefig(out/'price_split_scores.svg')
    svg=out/'price_split_scores.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.reports,a.out)
