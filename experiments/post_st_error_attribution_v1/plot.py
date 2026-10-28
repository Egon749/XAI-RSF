"""Reproducible aggregate research figure; no private company information."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def plot(run):
    out=run/'results'
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
    apparent=pd.read_csv(out/'training_apparent_metrics.csv',dtype={'seed':str})
    oob=pd.read_csv(out/'oob_metrics.csv',dtype={'seed':str})
    test=pd.read_csv(out/'metric_replay.csv',dtype={'scope':str,'seed':str})
    errors=pd.read_csv(out/'event_error_segments.csv',dtype={'scope':str,'seed':str})
    fig,(a,b)=plt.subplots(1,2,figsize=(12,5.7),gridspec_kw={'width_ratios':[1,1.15]})
    for year,color,marker in [(2023,'#2674a8','o'),(2024,'#cc6633','s')]:
        values=[apparent.query("test_year==@year and model=='equal' and seed=='ensemble' and cause=='termination'").auc.iloc[0],
                oob.query("test_year==@year and model=='equal' and seed=='ensemble' and cause=='termination'").auc.iloc[0],
                test.query("scope==@stryear and model=='equal' and seed=='ensemble' and cause=='termination'",local_dict={'stryear':str(year)}).auc.iloc[0]]
        a.plot(range(3),values,color=color,marker=marker,linewidth=2,label=f'{year}折')
        for x,y in enumerate(values):
            a.annotate(f'{y:.3f}',(x,y),xytext=(0,10 if year==2023 else -19),textcoords='offset points',ha='center',color=color)
    a.axhline(.5,color='#888888',ls=':',lw=1)
    a.set_xticks([0,1,2],['训练样本内','历史袋外','未来年度测试'])
    a.set_ylim(.45,1.055);a.set_xlim(-.2,2.2);a.set_ylabel('退市 ROC-AUC')
    a.set_title('历史内有信号，时间外表现下降',loc='left',pad=16)
    a.legend(loc='lower left',frameon=False);a.grid(axis='y',alpha=.18)
    g=errors.query("scope=='pooled' and model=='equal' and seed=='ensemble' and axis=='ground'").set_index('segment')
    keys=['price_below_one_yuan','financial_or_audit_ground','market_cap_threshold','failure_to_file_annual_report','not_explicit_in_available_terminal_text']
    labels=['低价退市','财务／审计','市值门槛','未按期披露年报','依据未明确']
    found=g.loc[keys,'captured_top20'].to_numpy();missed=g.loc[keys,'missed_top20'].to_numpy();y=np.arange(len(keys))
    b.barh(y,found,color='#2674a8',label='Top20%捕获')
    b.barh(y,missed,left=found,color='#d4d7db',hatch='//',label='未捕获')
    for yi,n,m in zip(y,found,missed):b.text(n+m+.35,yi,f'{int(n)}/{int(n+m)}',va='center')
    b.set_yticks(y,labels);b.invert_yaxis();b.set_xlim(0,23);b.set_xticks([0,5,10,15,20])
    b.set_xlabel('独立退市事件数');b.set_title('漏报涉及多种退市依据',loc='left',pad=16)
    b.legend(loc='lower right',frameon=False);b.grid(axis='x',alpha=.15);b.set_axisbelow(True)
    fig.suptitle('24个月退市预测：现有模型诊断',fontsize=16,x=.06,ha='left',y=.98)
    fig.text(.06,.055,'默认q2，三个固定种子等权集成；测试81家公司、31例退市。Top20%按各入组年度分别选取，共核查17家。',fontsize=10)
    fig.text(.06,.02,'袋外是混合历史年份的内部诊断，非时间外验证；本图不估计政策因果效果。全部结果均为探索性分析。',fontsize=10)
    fig.subplots_adjust(left=.065,right=.97,bottom=.2,top=.84,wspace=.43)
    for suffix in ['png','svg']:
        path=out/f'diagnostic_summary.{suffix}';assert not path.exists();fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);plot(p.parse_args().run)
