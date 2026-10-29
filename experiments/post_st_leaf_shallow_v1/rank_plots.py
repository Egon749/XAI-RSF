"""Plot all saved-model risk ranks; no fitting and no probability rescaling."""
from pathlib import Path
import argparse,datetime,json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions,endpoint_weights
from .analyze import artifacts

ARMS=['baseline','parent_shrink','shallow','shallow_parent']
NAMES={'baseline':'原基线（深度6）','parent_shrink':'深度6＋父节点收缩',
       'shallow':'深度4','shallow_parent':'深度4＋父节点收缩','standard_rsf':'标准RSF参照'}
RED='#CD4938';BLUE='#3B7D9C';GRAY='#9AA2AA';TOP20='#258577';TOP25='#9B6BB2'


def plot(run,out):
    assert out.is_dir() and not any(out.iterdir()),'Use a new empty private output directory'
    receipt=json.loads((run/'execution_receipt.json').read_text());assert receipt['status']=='COMPLETE'
    sources={};data={};counts=[];ranked=[];scales={}
    def checked(path):
        sources[str(path)]=sha(path);return path
    forecasts={(y,a,s):p for y,a,s,p in artifacts(run)}
    warning=pd.read_csv(checked(run/'evaluation/warning_metrics.csv'),dtype={'scope':str,'seed':str})
    metrics=pd.read_csv(checked(run/'evaluation/metrics.csv'),dtype={'scope':str,'seed':str})
    for year in [2023,2024]:
        ev=arrays(checked(run/'evaluation_input'/f'{year}_test_PRIVATE.npz'))
        _,known,_=endpoint_weights(ev['analysis_time'],ev['event'],24.,ev['label_eligible'])
        assert known.all() and len(np.unique(ev['groups']))==len(known)
        target=(ev['event']==1)&(ev['analysis_time']<=24)
        recovery=(ev['event']==2)&(ev['analysis_time']<=24)
        assert target.sum()=={2023:12,2024:20}[year]
        for arm in ARMS+['standard_rsf']:
            ps=[]
            for seed in ['42','101','202']:
                folder=forecasts[(year,arm,seed)]
                hashes=json.loads((folder/'hashes.json').read_text())
                path=checked(folder/'predictions_PRIVATE.npz');assert sha(path)==hashes[path.name]
                p=arrays(path);np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids'])
                ps.append(horizon_predictions(p['cif'],p['times'],ev['h24'])[:,0])
            risk=np.mean(ps,axis=0)
            assert np.isfinite(risk).all() and risk.min()>=0 and risk.max()<=1
            order=np.lexsort((ev['sample_ids'],-risk));assert np.all(np.diff(risk[order])<=0)
            n=len(order);q={}
            for frac in [.2,.25]:
                k=int(np.ceil(n*frac));hit=int(target[order[:k]].sum())
                row=warning[(warning.scope==str(year))&(warning.arm==arm)&(warning.seed=='ensemble')&
                    (warning.horizon_months==24)&(warning.top_fraction==frac)]
                assert len(row)==1 and row.iloc[0].checked==k and row.iloc[0].captured==hit and row.iloc[0].unknown==0
                q[frac]=(k,hit)
                counts.append(dict(year=year,arm=arm,companies=n,total_target_events=int(target.sum()),top_fraction=frac,
                    checked=k,captured=hit,missed=int(target.sum())-hit,coverage=hit/int(target.sum()),
                    selected_known_non_target=k-hit,unknown=0))
            mm=metrics[(metrics.scope==str(year))&(metrics.arm==arm)&(metrics.seed=='ensemble')&
                (metrics.cause=='termination')&(metrics.horizon_months==24)].iloc[0]
            data[(year,arm)]=dict(n=n,rank=np.arange(1,n+1),risk=risk[order],target=target[order],recovery=recovery[order],
                targets=int(target.sum()),selected=q,auc=mm.auc,ap=mm.ap)
            for rank,ix in enumerate(order,1):
                ranked.append(dict(year=year,arm=arm,rank=rank,sample_id=ev['sample_ids'][ix],risk=float(risk[ix]),
                    target=bool(target[ix]),recovery=bool(recovery[ix]),top20=rank<=q[.2][0],top25=rank<=q[.25][0]))
        scales[year]=max(.05,math.ceil(max(data[(year,a)]['risk'].max() for a in ARMS+['standard_rsf'])/.05)*.05)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#87929D',
        'text.color':'#263544','axes.labelcolor':'#263544','xtick.color':'#53616E','ytick.color':'#53616E',
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    def panel(ax,year,arm):
        d=data[(year,arm)];n=d['n'];r=d['rank'];risk=d['risk'];k20,h20=d['selected'][.2];k25,h25=d['selected'][.25]
        ax.axvspan(.5,k20+.5,color=TOP20,alpha=.075,zorder=0)
        ax.axvspan(k20+.5,k25+.5,color=TOP25,alpha=.055,zorder=0)
        ax.axvline(k20+.5,color=TOP20,linewidth=1.5,zorder=1)
        ax.axvline(k25+.5,color=TOP25,linewidth=1.5,linestyle=(0,(4,3)),zorder=1)
        ax.plot(r,risk,color='#B9C2CC',linewidth=1.1,zorder=2)
        other=~(d['target']|d['recovery'])
        ax.scatter(r[other],risk[other],facecolors='white',edgecolors=GRAY,s=35,linewidths=1.2,zorder=3)
        ax.scatter(r[d['recovery']],risk[d['recovery']],color=BLUE,s=36,edgecolor='white',linewidths=.6,zorder=4)
        ax.scatter(r[d['target']],risk[d['target']],color=RED,s=51,edgecolor='white',linewidths=.7,zorder=5)
        ax.set_xlim(.5,n+.5);ax.set_ylim(0,scales[year])
        ax.set_xticks(sorted(set([1,n]+[v for v in range(5,n,5) if n-v>=3])))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis='y',color='#E4E9EE',linewidth=.8)
        ax.set_axisbelow(True);ax.set_xlabel('风险排名（左高 → 右低）');ax.set_ylabel('预测24个月先退市概率')
        ax.set_title(NAMES[arm]+f'   |   AUC {d["auc"]:.3f} · AP {d["ap"]:.3f}',loc='left',fontsize=12,pad=13)
        ax.text(.98,.96,f'Top20%：查{k20}家，捕获{h20}/{d["targets"]}例\nTop25%：查{k25}家，捕获{h25}/{d["targets"]}例',
            transform=ax.transAxes,ha='right',va='top',fontsize=10.6,linespacing=1.5,
            bbox=dict(boxstyle='round,pad=.4',facecolor='white',edgecolor='#DEE4EA',alpha=.94))
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=RED,markeredgecolor='white',markersize=8,label='24个月内先退市'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor=BLUE,markeredgecolor='white',markersize=7,label='24个月内先恢复'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor='white',markeredgecolor=GRAY,markersize=6,label='24个月内无首次事件'),
        Line2D([0],[0],color=TOP20,lw=1.6,label='Top20% 边界'),Line2D([0],[0],color=TOP25,lw=1.6,linestyle='--',label='Top25% 边界')]
    with PdfPages(out/'risk_rankings_2023_2024_PRIVATE.pdf') as pdf:
        for year in [2023,2024]:
            n=data[(year,'baseline')]['n'];targets=data[(year,'baseline')]['targets']
            fig,axs=plt.subplots(2,2,figsize=(14,8.8))
            fig.subplots_adjust(left=.075,right=.975,bottom=.18,top=.83,wspace=.22,hspace=.43)
            fig.suptitle(f'{year}年｜全部{n}家企业的风险排序',fontsize=21,y=.975,fontweight='bold')
            fig.text(.5,.915,f'观察到{targets}例先退市；每个面板独立排序，红点越集中在左侧，前排事件捕获越好',ha='center',fontsize=12,color='#53616E')
            for ax,arm in zip(axs.flat,ARMS):panel(ax,year,arm)
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=5,frameon=False,fontsize=10.5)
            fig.text(.5,.04,'原始预测概率；三个固定种子平均。横轴是排名，同一排名在不同模型中未必对应同一家企业。',ha='center',fontsize=10,color='#65717B')
            fig.text(.5,.012,'核查人数向上取整；全部结局已知。24个日历月、公告日终点；仅使用现有模型。',ha='center',fontsize=9.5,color='#65717B')
            fig.savefig(out/f'risk_ranking_{year}_PRIVATE.png',dpi=180)
            fig.savefig(out/f'risk_ranking_{year}_PRIVATE.pdf')
            pdf.savefig(fig);plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(14,4.9))
        fig.subplots_adjust(left=.075,right=.975,bottom=.27,top=.76,wspace=.22)
        fig.suptitle('标准RSF参照｜全部企业风险排序',fontsize=20,y=.97)
        for ax,year in zip(axs,[2023,2024]):
            panel(ax,year,'standard_rsf')
            d=data[(year,'standard_rsf')]
            ax.set_title(f'{year}年 · {d["n"]}家企业   |   AUC {d["auc"]:.3f} · AP {d["ap"]:.3f}',loc='left',fontsize=12,pad=13)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.1),ncol=5,frameon=False,fontsize=10.5)
        fig.text(.5,.035,'本ST后任务的冻结标准RSF；相同年度纵轴与q2图一致。企业全部显示，未重新训练。',ha='center',fontsize=10,color='#65717B')
        fig.savefig(out/'standard_rsf_rankings_PRIVATE.png',dpi=180);pdf.savefig(fig);plt.close(fig)
    frame=pd.DataFrame(counts)
    extra=[]
    for (arm,frac),g in frame.groupby(['arm','top_fraction']):
        extra.append(dict(year='pooled',arm=arm,companies=int(g.companies.sum()),total_target_events=int(g.total_target_events.sum()),
            top_fraction=frac,checked=int(g.checked.sum()),captured=int(g.captured.sum()),missed=int(g.missed.sum()),
            coverage=g.captured.sum()/g.total_target_events.sum(),selected_known_non_target=int(g.selected_known_non_target.sum()),unknown=0))
    frame=pd.concat([frame,pd.DataFrame(extra)],ignore_index=True);frame.to_csv(out/'capture_counts.csv',index=False)
    pd.DataFrame(ranked).to_csv(out/'ranked_predictions_PRIVATE.csv',index=False)
    for path,h in sources.items():assert sha(Path(path))==h
    write(out/'plot_receipt.json',dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_fitting_commit=receipt['fitting_commit'],source_run=str(run),new_fits=0,
        horizon_calendar_months=24,prediction='mean over seeds42/101/202; native CIF, not min-max scaled',
        displayed_years=[2023,2024],companies_by_year={'2023':36,'2024':45},events_by_year={'2023':12,'2024':20},
        panels=10,private_rank_rows=len(ranked),all_companies_displayed=True,all20_25_counts_match_published_results=True,
        tie_break='sample_id lexical; IDs absent from figures',renderer='matplotlib '+matplotlib.__version__,
        renderer_reason='Scientific static PNG/PDF; Data scatter grouped outcome colors cannot combine with reference lines (documented unsupported encoding).',
        privacy='Figures and per-company ranks remain server-private; code and aggregate counts may be published',
        source_sha256=sources,script_sha256=sha(Path(__file__)),
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(frame.to_string(index=False));print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.run,a.output)
