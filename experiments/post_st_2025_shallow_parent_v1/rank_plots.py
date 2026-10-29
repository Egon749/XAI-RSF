"""Plot all 2025 companies from the six saved forecasts; never fit or rescale."""
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
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions,endpoint_weights
from experiments.post_st_2025_shallow_parent_v1.analyze import outputs


def plot(run,out):
    assert out.is_dir() and not any(out.iterdir()),'Use a new empty private directory'
    receipt=json.loads((run/'execution_receipt.json').read_text());assert receipt['status']=='COMPLETE'
    sources={}
    def source(p,expected=None):
        h=sha(p)
        if expected is not None:assert h==expected,str(p)
        sources[str(p)]=h
        return p
    ev=arrays(source(run/'evaluation_input/2025_test_PRIVATE.npz'))
    w,known,_=endpoint_weights(ev['analysis_time'],ev['event'],12.,ev['label_eligible'])
    assert known.all();np.testing.assert_array_equal(w,np.ones(len(w)))
    target=(ev['event']==1)&(ev['analysis_time']<=12)
    recovery=(ev['event']==2)&(ev['analysis_time']<=12)
    assert len(w)==53 and target.sum()==5 and recovery.sum()==3
    metrics=pd.read_csv(source(run/'evaluation/metrics.csv'),dtype={'seed':str})
    warnings=pd.read_csv(source(run/'evaluation/warning_metrics.csv'),dtype={'seed':str})
    prior=pd.read_csv(source(run/'private/ranked_predictions_PRIVATE.csv'))
    parts={}
    for model,seed,folder in outputs(run):
        hashes=json.loads(source(folder/'hashes.json').read_text())
        p=arrays(source(folder/'predictions_PRIVATE.npz',hashes['predictions_PRIVATE.npz']))
        np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids'])
        parts[(model,seed)]=horizon_predictions(p['cif'],p['times'],ev['h12'])[:,0]
    data={};summaries=[];rows=[]
    for model in ['xai_q2','standard_rsf']:
        risk=np.mean([parts[(model,str(s))] for s in [42,101,202]],axis=0)
        assert np.isfinite(risk).all() and risk.min()>=0 and risk.max()<=1
        order=np.lexsort((ev['sample_ids'],-risk));rank=np.arange(1,len(order)+1)
        assert (np.diff(risk[order])<=0).all()
        saved=prior[prior.model.eq(model)].sort_values('rank')
        np.testing.assert_array_equal(saved['rank'],rank)
        np.testing.assert_array_equal(saved.sample_id,ev['sample_ids'][order])
        np.testing.assert_allclose(saved.risk,risk[order],atol=2e-16,rtol=0)
        np.testing.assert_array_equal(saved.target,target[order])
        m=metrics[metrics.model.eq(model)&metrics.seed.eq('ensemble')&metrics.horizon_months.eq(12)&metrics.cause.eq('termination')].iloc[0]
        selected=warnings[warnings.model.eq(model)&warnings.seed.eq('ensemble')&warnings.horizon_months.eq(12)&warnings.top_fraction.eq(.2)].iloc[0]
        ncheck=int(np.ceil(.2*len(order)));hit=int(target[order[:ncheck]].sum())
        assert ncheck==selected.checked and hit==selected.captured and selected.unknown==0
        d=dict(model=model,companies=len(order),target_events=int(target.sum()),recovery_events=int(recovery.sum()),
            known_no_first_event=int((~(target|recovery)).sum()),unknown=0,top20_checked=ncheck,top20_captured=hit,
            top20_missed=int(target.sum())-hit,auc=float(m.auc),ap=float(m.ap),risk_min=float(risk.min()),risk_max=float(risk.max()))
        summaries.append(d)
        data[model]={**d,'rank':rank,'risk':risk[order],'target':target[order],'recovery':recovery[order]}
        for r,i in enumerate(order,1):rows.append(dict(model=model,rank=r,sample_id=ev['sample_ids'][i],risk=float(risk[i]),
            target=bool(target[i]),recovery=bool(recovery[i]),known=bool(known[i]),top20=r<=ncheck))
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#83929E',
        'text.color':'#243442','axes.labelcolor':'#243442','xtick.color':'#53616D','ytick.color':'#53616D',
        'pdf.fonttype':42,'savefig.facecolor':'white'})
    red='#CC4939';blue='#367FAA';gray='#939DA7';green='#20836D'
    ymax=max(.05,math.ceil(max(d['risk_max'] for d in data.values())/.05)*.05)
    fig,axs=plt.subplots(1,2,figsize=(14.4,6.3),sharey=True)
    fig.subplots_adjust(left=.075,right=.98,bottom=.27,top=.74,wspace=.17)
    fig.suptitle('2025年｜全部53家企业的退市风险排序',fontsize=21,y=.97,fontweight='bold')
    fig.text(.5,.906,'预测入组后12个月先退市；观察截止2026-06-30；三个固定种子的平均预测',ha='center',fontsize=11.5,color='#53616D')
    names={'xai_q2':'我们的模型：深度4＋收缩','standard_rsf':'普通RSF'}
    for ax,model in zip(axs,['xai_q2','standard_rsf']):
        d=data[model];rank=d['rank'];risk=d['risk'];k=d['top20_checked'];other=~(d['target']|d['recovery'])
        ax.axvspan(.5,k+.5,color=green,alpha=.07,zorder=0)
        ax.axvline(k+.5,color=green,lw=1.5,zorder=1)
        ax.plot(rank,risk,color='#C3CBD2',lw=1.1,zorder=2)
        ax.scatter(rank[other],risk[other],facecolors='white',edgecolors=gray,s=39,lw=1.15,zorder=3)
        ax.scatter(rank[d['recovery']],risk[d['recovery']],color=blue,marker='s',s=43,edgecolors='white',lw=.6,zorder=4)
        ax.scatter(rank[d['target']],risk[d['target']],color=red,s=76,edgecolors='white',lw=.8,zorder=5)
        ax.set_xlim(.5,53.5);ax.set_ylim(-ymax*.025,ymax)
        ax.set_xticks([1,11,20,30,40,53]);ax.set_yticks(np.arange(0,ymax+.001,.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.tick_params(labelleft=True)
        ax.grid(axis='y',color='#E5E9EC',lw=.8);ax.set_axisbelow(True)
        ax.set_xlabel('预测风险排名（左高 → 右低）',labelpad=10)
        ax.set_title(names[model]+f'\nAUC {d["auc"]:.3f}  ·  AP {d["ap"]:.3f}',loc='left',fontsize=13,pad=13,linespacing=1.6)
        ax.text(.97,.95,f'前20%：核查{k}家\n捕获{d["top20_captured"]}/5例退市（{d["top20_captured"]/5:.0%}）',
            transform=ax.transAxes,ha='right',va='top',fontsize=11,linespacing=1.6,
            bbox=dict(boxstyle='round,pad=.45',facecolor='white',edgecolor='#DCE3E7',alpha=.96))
    axs[0].set_ylabel('预测12个月先退市概率',labelpad=10)
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=red,markeredgecolor='white',markersize=9,label='12个月内先退市（5家）'),
        Line2D([0],[0],marker='s',color='none',markerfacecolor=blue,markeredgecolor='white',markersize=7,label='12个月内先恢复（3家）'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor='white',markeredgecolor=gray,markersize=7,label='12个月内无首次结局（45家）'),
        Line2D([0],[0],color=green,lw=1.6,label='前20%边界（前11家）')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.13),ncol=4,frameon=False,fontsize=10.3)
    fig.text(.5,.08,'每个点是一家公司；两图分别排序，同一名次未必是同一家。纵轴尺度相同，保留原始概率，未做0—1拉伸。',ha='center',fontsize=10,color='#64717C')
    fig.text(.5,.035,'所有53家公司均显示；红点越集中在绿色区域，前20%的退市事件捕获越好。此图不评价24个月结局。',ha='center',fontsize=10,color='#64717C')
    fig.savefig(out/'risk_rankings_2025_q2_vs_rsf_PRIVATE.png',dpi=180)
    fig.savefig(out/'risk_rankings_2025_q2_vs_rsf_PRIVATE.pdf');plt.close(fig)
    pd.DataFrame(summaries).to_csv(out/'ranking_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'ranked_predictions_PRIVATE.csv',index=False)
    assert all(sha(p)==h for p,h in sources.items())
    write(out/'plot_receipt.json',dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=str(run),source_fitting_commit=receipt['fitting_commit'],new_fits=0,new_predictions=0,
        panels=2,companies_per_panel=53,positive_events_per_panel=5,horizon_calendar_months=12,
        observation_cutoff='2026-06-30',all_companies_displayed=True,all_metrics_and_ranks_match_saved_outputs=True,
        common_y_upper=ymax,probability_rescaling=False,tie_break='sample_id lexical; IDs absent from figures',
        renderer='matplotlib '+matplotlib.__version__,
        renderer_reason='Scientific static PNG/PDF; Data grouped scatter cannot also contain reference lines/bands.',
        privacy='Individual dots and ranking files remain server-private; publish code and aggregate summaries only.',
        source_sha256=sources,script_sha256=sha(Path(__file__)),output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(pd.DataFrame(summaries).to_string(index=False));print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.run,a.output)
