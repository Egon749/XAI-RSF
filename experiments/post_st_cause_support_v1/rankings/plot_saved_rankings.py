"""Private saved-prediction risk rankings. No fitting or prediction changes."""
from pathlib import Path
import argparse, json, math, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_cause_support_v1.common import arrays,sha,horizon_predictions,TIMES
from experiments.post_st_feature_permutation_v1.common import evaluation_context

def plot_saved_rankings(run, output):
    RUN=Path(run);OUT=Path(output)
    OUT.mkdir(parents=True,exist_ok=False)
    source_hashes={}
    def source(path):
        source_hashes[str(path)]=sha(path)
        return path

    registry=json.loads(source(RUN/'audit/registry_formal.json').read_text())
    metrics=pd.read_csv(source(RUN/'reports/METRICS_ENSEMBLE.csv'))
    summaries=[];rows=[];data={}
    for year in [2023,2024]:
        lab=arrays(source(RUN/f'evaluation_input/{year}_test_PRIVATE.npz'))
        assert len(np.unique(lab['groups']))==len(lab['groups'])
        records=sorted([r for r in registry if r['year']==year and r['arm']=='support'],key=lambda r:r['seed'])
        assert [r['seed'] for r in records]==[42,101,202]
        predictions=[]
        for r in records:
            root=Path(r['path']);hashes=json.loads(source(root/'hashes.json').read_text())
            p=root/'shrunk/predictions_PRIVATE.npz';assert sha(p)==hashes['shrunk/predictions_PRIVATE.npz']
            z=arrays(source(p));np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
            predictions.append(z['cif'])
        mean=np.mean(predictions,axis=0)
        for horizon in [12,24]:
            risk=horizon_predictions(mean,TIMES,lab[f'h{horizon}'])[:,0]
            event,weights,known,supported,_=evaluation_context(lab,horizon,1)
            assert supported and known.all();np.testing.assert_array_equal(weights,np.ones(len(risk)))
            recovery=(lab['event']==2)&(lab['analysis_time']<=horizon)
            assert not (event&recovery).any()
            order=np.lexsort((lab['sample_ids'],-risk));n=len(order);count=int(np.ceil(.2*n));hits=int(event[order[:count]].sum())
            assert np.isfinite(risk).all() and risk.min()>=0 and risk.max()<=1 and np.all(np.diff(risk[order])<=0)
            m=metrics[(metrics.scope==str(year))&(metrics.arm=='support')&(metrics.version=='shrunk')&(metrics.months==horizon)&(metrics.cause==1)]
            assert len(m)==1;m=m.iloc[0]
            auc=roc_auc_score(event,risk);ap=average_precision_score(event,risk)
            np.testing.assert_allclose([auc,ap,float(np.mean((risk-event)**2)),float(risk.mean())],[m.auc,m.ap,m.brier,m.risk_mean],atol=1e-13,rtol=0)
            assert hits==m.captured and event.sum()==m.target_events and count==m.checked
            summary=dict(year=year,months=horizon,arm='support',version='shrunk',companies=n,termination=int(event.sum()),recovery=int(recovery.sum()),
                top20_companies=count,top20_captured=hits,coverage=hits/int(event.sum()),top20_precision=hits/count,
                unknown=0,auc=auc,ap=ap,brier=float(m.brier),max_risk=float(risk.max()),mean_risk=float(risk.mean()),
                target_ranks=(np.flatnonzero(event[order])+1).tolist())
            summaries.append(summary);data[(year,horizon)]=dict(**summary,rank=np.arange(1,n+1),risk=risk[order],event=event[order],recovered=recovery[order])
            for rank,ix in enumerate(order,1):
                rows.append(dict(year=year,months=horizon,rank=rank,sample_id=lab['sample_ids'][ix],risk=float(risk[ix]),
                    termination=bool(event[ix]),recovery=bool(recovery[ix]),known=True,top20=rank<=count))

    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#89959d','text.color':'#253743','axes.labelcolor':'#253743',
        'xtick.color':'#526471','ytick.color':'#526471','pdf.fonttype':42,'savefig.facecolor':'white'})
    red='#d1493d';blue='#4087b2';gray='#a2abb3';green='#347e69'
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=red,markeredgecolor='white',markersize=8,label='时限内先最终退市'),
        Line2D([0],[0],marker='^',color='none',markerfacecolor=blue,markeredgecolor='white',markersize=8,label='时限内先恢复'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor='white',markeredgecolor=gray,markersize=7,label='时限内无首次事件'),
        Line2D([0],[0],color=green,linestyle='--',lw=1.6,label='前20%核查边界（人数向上取整）')]

    with PdfPages(OUT/'support_risk_rankings_2023_2024_PRIVATE.pdf') as pdf:
        for horizon in [12,24]:
            fig,axes=plt.subplots(1,2,figsize=(13.8,5.4));fig.subplots_adjust(left=.07,right=.98,top=.76,bottom=.27,wspace=.22)
            upper=math.ceil(max(data[(year,horizon)]['max_risk'] for year in [2023,2024])/.01)*.01
            upper=max(upper,.06 if horizon==12 else .20)
            for ax,year in zip(axes,[2023,2024]):
                d=data[(year,horizon)];r=d['rank'];p=d['risk'];k=d['top20_companies'];hit=d['top20_captured'];other=~(d['event']|d['recovered'])
                ax.axvspan(.5,k+.5,color=green,alpha=.08,zorder=0);ax.axvline(k+.5,color=green,ls='--',lw=1.4,zorder=1)
                ax.plot(r,p,color='#c2cbd2',lw=1,zorder=2)
                ax.scatter(r[other],p[other],s=43,facecolors='white',edgecolors=gray,lw=1.2,zorder=3)
                ax.scatter(r[d['recovered']],p[d['recovered']],marker='^',s=58,color=blue,edgecolors='white',lw=.7,zorder=4)
                ax.scatter(r[d['event']],p[d['event']],s=62,color=red,edgecolors='white',lw=.7,zorder=5)
                ax.set_xlim(.5,d['companies']+.5);ax.set_ylim(0,upper)
                ax.set_xticks([1,*[k for k in range(5,d['companies'],5) if d['companies']-k>=3],d['companies']])
                ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_yticks(np.linspace(0,upper,7 if horizon==12 else 6))
                ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
                ax.set_xlabel('企业风险排名（左高 → 右低）',labelpad=9);ax.set_ylabel(f'预测{horizon}个月先退市概率')
                ax.set_title(f'{year}年 · 全部{d["companies"]}家企业\nAUC {d["auc"]:.3f}   AP {d["ap"]:.3f}',loc='left',fontsize=13,pad=14)
                ax.text(.98,.95,f'前20%：核查{k}家\n捕获{hit}/{d["termination"]}例退市（{d["coverage"]:.1%}）',transform=ax.transAxes,
                    ha='right',va='top',fontsize=11,linespacing=1.6,bbox=dict(boxstyle='round,pad=.4',facecolor='white',edgecolor='#e0e6e9',alpha=.97))
                # Numeric labels identify event ranks without exposing company identity.
                for rank in d['target_ranks']:
                    if year==2023 or rank>=d['companies']-4:
                        ax.annotate(str(rank),(rank,p[rank-1]),xytext=(0,8),textcoords='offset points',ha='center',va='bottom',color=red,fontsize=9)
            fig.suptitle(f'退市支持保护｜{horizon}个月风险排序',fontsize=21,fontweight='bold',y=.98)
            fig.text(.5,.88,'每个点是一家企业；红点越集中在左侧，退市企业的排序越靠前',ha='center',fontsize=12,color='#526471')
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.10),ncol=4,frameon=False,fontsize=10)
            fig.text(.5,.055,'83变量 · q2 · 深度4 · 父节点收缩；仅加入退市支持保护。三个种子的已保存预测取平均，未重新训练。',ha='center',fontsize=10,color='#526471')
            fig.text(.5,.015,'每个年度、时限独立排序；纵轴为原始CIF概率，未放大或重新映射。非红点不等于企业以后不会退市。',ha='center',fontsize=9.5,color='#526471')
            fig.savefig(OUT/f'support_risk_ranking_{horizon}m_PRIVATE.png',dpi=180)
            fig.savefig(OUT/f'support_risk_ranking_{horizon}m_PRIVATE.pdf');pdf.savefig(fig);plt.close(fig)

    pd.DataFrame(rows).to_csv(OUT/'ranked_companies_PRIVATE.csv',index=False)
    pd.DataFrame([{k:v for k,v in s.items() if k!='target_ranks'} for s in summaries]).to_csv(OUT/'rank_summary.csv',index=False)
    (OUT/'event_ranks_PRIVATE.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    for p,h in source_hashes.items():assert sha(Path(p))==h
    receipt=dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_fits=0,source_run=str(RUN),
        arm='support only, unchanged signed_sum score',model_version='shrunk',seeds=[42,101,202],years=[2023,2024],horizons_calendar_months=[12,24],
        prediction='mean seed CIF then calendar-horizon extraction; no rescaling',all_companies_shown=True,all_known=True,
        tied_risk_order='sample_id lexicographic',full_match_to_published_metrics=True,
        privacy='Private by default; publication requires the recorded researcher authorization; no company identifiers in figures',
        renderer='matplotlib static scientific export; Data scatter grouped outcomes and reference boundary combination is unsupported',
        sources_sha256=source_hashes,script_sha256=sha(Path(__file__)),outputs_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='plot_receipt.json' and p.suffix!='.log'})
    (OUT/'plot_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(summaries,ensure_ascii=False,indent=2));print(OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Plot saved support-arm forecasts only; never fit.')
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();plot_saved_rankings(args.run,args.output)
