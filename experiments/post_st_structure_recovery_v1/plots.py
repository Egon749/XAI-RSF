"""Exportable research figures; all displayed functions concern synthetic data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from crforest import HigherOrderCIFExpansion
from .common import *

LABELS={
 'additive_small':'纯加性 · 350', 'pair_small':'二阶 · 350','pair_large':'二阶 · 3000',
 'pure_pair_large':'纯交互 · 3000','rare_pair_small':'稀少事件 · 350',
 'correlated_pair_large':'相关变量 · 3000','heavy_censor_pair_large':'较强删失 · 3000',
 'triple_large':'含真实三阶 · 3000','continuous_pair_large':'连续变量 · 3000'}

def setup():
    fonts=font_manager.findSystemFonts()
    chosen=next((f for f in fonts if 'NotoSansCJK-Regular' in f),None)
    if chosen:font_manager.fontManager.addfont(chosen);plt.rcParams['font.family']=font_manager.FontProperties(fname=chosen).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':190})

def save(fig,folder,name):
    fig.savefig(folder/f'{name}.png',bbox_inches='tight');fig.savefig(folder/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def plots(run,summary,signal):
    setup();out=run/'reports';cases=[s for s in SCENARIOS if s!='additive_small']
    colors={'q2':'#2166ac','unrestricted':'#ce5232'}
    ss=signal[(signal.seed=='ensemble')&(signal.day==730)&(signal.cause==1)&(signal.role=='true_pair')&(signal.metric=='amplitude_ratio')]
    fig,ax=plt.subplots(figsize=(10,6));y=np.arange(len(cases))
    for offset,(arm,variant,style) in zip([-.24,-.08,.08,.24],[('q2','raw','o'),('q2','shrunk','s'),('unrestricted','raw','o'),('unrestricted','shrunk','s')]):
        g=ss[(ss.arm==arm)&(ss.variant==variant)].set_index('scenario').reindex(cases)
        ax.errorbar(g['mean']*100,y+offset,xerr=np.vstack([(g['mean']-g.low95)*100,(g.high95-g['mean'])*100]),fmt=style,color=colors[arm],
            mfc='white' if variant=='raw' else colors[arm],capsize=2,label=f"{'q2' if arm=='q2' else '不限q'} · {'原始AJ' if variant=='raw' else '父节点收缩'}")
    ax.axvline(100,color='#555555',ls='--',lw=1,label='真实幅度 = 100%');ax.axvline(0,color='#aaaaaa',lw=.6)
    ax.set(yticks=y,yticklabels=[LABELS[s] for s in cases],xlabel='真实二阶交互的幅度恢复比例（%）',title='精确解释与学对真实作用，是两个不同问题')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(loc='upper right',fontsize=9)
    fig.text(.12,-.015,'点为50份独立数据的均值；横线为均值的95% Monte Carlo区间。三种子先合并为96棵树。\n连续变量为已知乘积项的投影系数；完整函数误差另行报告。',fontsize=9)
    save(fig,out,'interaction_amplitude_recovery')
    primary=summary[(summary.seed=='ensemble')&(summary.variant=='shrunk')&(summary.day==730)&(summary.cause==1)]
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True);cases_all=list(SCENARIOS);y=np.arange(len(cases_all))
    for ax,metric,title,scale in [(axes[0],'pair_recall','真实二阶交互检出率',100),(axes[1],'all_fp','额外检出的无关函数数',1)]:
        for off,arm in [(-.12,'q2'),(.12,'unrestricted')]:
            g=primary[(primary.arm==arm)&(primary.metric==metric)].set_index('scenario').reindex(cases_all)
            lo=g.rate_wilson_low95 if metric=='pair_recall' else g.low95
            hi=g.rate_wilson_high95 if metric=='pair_recall' else g.high95
            ax.errorbar(g['mean']*scale,y+off,xerr=np.maximum(0,np.vstack([(g['mean']-lo)*scale,(hi-g['mean'])*scale])),fmt='o',color=colors[arm],capsize=2,label='q2' if arm=='q2' else '不限q')
        ax.set(title=title,xlabel='%' if scale==100 else '个');ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].set(yticks=y,yticklabels=[LABELS[s] for s in cases_all]);axes[0].invert_yaxis()
    fig.text(.12,-.015,'RMS≥0.5个百分点记为检出；检出率为Wilson区间，无关函数数为均值的t区间。\n纯加性场景没有真实交互，交互检出率不定义；无关函数包括任意阶数。',fontsize=9)
    save(fig,out,'structure_recovery')
    fig,axes=plt.subplots(1,2,figsize=(12,5.8),sharey=True)
    for ax,metric,title in [(axes[0],'joint_risk_mse','真实联合分布上的CIF均方误差'),(axes[1],'spatial_cells_including_missing','非基线函数表空间单元数')]:
        for off,arm in [(-.12,'q2'),(.12,'unrestricted')]:
            g=primary[(primary.arm==arm)&(primary.metric==metric)].set_index('scenario').reindex(cases_all)
            ax.errorbar(g['mean'],y+off,xerr=np.vstack([g['mean']-g.low95,g.high95-g['mean']]),fmt='o',color=colors[arm],capsize=2,label='q2' if arm=='q2' else '不限q')
        ax.set(title=title);ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].set(yticks=y,yticklabels=[LABELS[s] for s in cases_all]);axes[0].invert_yaxis()
    fig.text(.12,-.015,'均使用强度10父节点收缩；预测误差比较已知真实概率，未把删失样本当作未发生事件。\n空间单元含独立缺失状态，不乘时间或原因维度；本轮输入没有缺失。',fontsize=9)
    save(fig,out,'prediction_complexity')
    # Fixed example: repetition zero, chosen BEFORE reading estimates.
    fig,axes=plt.subplots(2,3,figsize=(11,6.6));job=json.loads((run/'audit/formal_gate.json').read_text())['job']
    for row,scenario in enumerate(['pair_large','continuous_pair_large']):
        idx=list(SCENARIOS).index(scenario);folder=run/'results/formal'/str(job)/f'task-{idx}/output'
        key=next(k for k in truth_terms(scenario,0) if len(k)==2)
        z=np.array([-1.,1.]) if row==0 else np.linspace(-1,1,61);a,b=np.meshgrid(z,z);X=np.zeros((a.size,P));X[:,key[0]]=a.ravel();X[:,key[1]]=b.ravel()
        values=[.05*a*b]
        for arm in ARMS:
            ex=HigherOrderCIFExpansion.load(folder/f'{arm}_ensemble_shrunk/population_functions.npz')
            values.append(ex.component(X,key,[730])[:,0,0].reshape(a.shape))
        for col,v in enumerate(values):
            im=axes[row,col].imshow(v*100,origin='lower',extent=[-1,1,-1,1],cmap='RdBu_r',vmin=-5,vmax=5,interpolation='nearest',aspect='auto')
            axes[row,col].set(xlabel='真实交互变量A',ylabel='真实交互变量B',title=f"{['已知真值','q2＋收缩','不限q＋收缩'][col]} · {'二元' if row==0 else '连续'}")
            if row==0:
                for iy,yv in enumerate([-.5,.5]):
                    for ix,xv in enumerate([-.5,.5]):axes[row,col].text(xv,yv,f'{v[iy,ix]*100:+.2f}',ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.6,edgecolor='none'))
    fig.subplots_adjust(right=.88,hspace=.5,wspace=.4);bar=fig.add_axes([.91,.2,.02,.6]);fig.colorbar(im,cax=bar,label='24个月CIF贡献（百分点）')
    fig.text(.1,.015,'固定展示第0份模拟数据，非择优案例；两模型使用同一训练数据，贡献来自完整函数表。',fontsize=9)
    save(fig,out,'fixed_true_pair_functions')
