"""Display saved local contributions as direction + absolute share; no fit.

Shares use all main and pair terms together, exclude the reference baseline,
and never allocate an interaction to its individual input variables.
"""
from pathlib import Path
import argparse,datetime,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiments.post_st_prognosis_v1.common import arrays,sha,write


def run(source,receipt,reference,output):
    output.mkdir(parents=True,exist_ok=False)
    record=json.loads(receipt.read_text());current=json.loads(reference.read_text())
    assert current['experiment']=='post_st_leaf_shallow_v1' and current['arm']=='shallow_parent'
    assert current['configuration']['case_time_weight']==1 and current['configuration']['min_leaf']==10
    assert Path(record['private_output_directory']).resolve()==source.resolve()
    sources={str(receipt):sha(receipt),str(reference):sha(reference)}
    for name,h in record['output_sha256'].items():
        assert sha(source/name)==h,name
        sources[str(source/name)]=h
    original=json.loads((source/'execution_receipt.json').read_text())
    for name,h in original['source_sha256'].items():
        assert sha(Path(name))==h,name
        sources[name]=h
    for model in current['saved_models']:
        if model['test_year']==2024:
            for key in ['forest_sha256','predictions_sha256','functions_sha256']:
                assert model[key] in original['source_sha256'].values()
    info=json.loads((source/'case_summary_PRIVATE.json').read_text())
    frame=pd.read_csv(source/'all_contributions_PRIVATE.csv')
    curve=arrays(source/'full_curve_reconstruction_PRIVATE.npz')
    assert len(frame)==229 and frame.term_id.is_unique and (frame.order==1).sum()==83 and (frame.order==2).sum()==146
    pos=int(np.flatnonzero(curve['times']==info['horizon_days'])[0])
    pp=frame.termination_pp.to_numpy();total=float(abs(pp).sum())
    assert total>1e-12,'No nonzero contributions: percentage shares are undefined'
    np.testing.assert_allclose(pp,100*curve['terms'][:,pos,0],atol=1e-12,rtol=0)
    np.testing.assert_allclose(curve['baseline']+curve['terms'].sum(axis=0),curve['model'],atol=1e-12,rtol=0)
    frame['direction']=np.where(pp>0,'升高',np.where(pp<0,'降低','零'))
    frame['absolute_share_percent']=100*abs(pp)/total
    frame['signed_share_percent']=100*pp/total
    assert abs(frame.absolute_share_percent.sum()-100)<1e-12
    back=total*frame.signed_share_percent.to_numpy()/100
    np.testing.assert_allclose(back,pp,atol=1e-12,rtol=0)
    baseline=float(info['base_percent']);risk=float(info['predicted_percent'])
    reconstruction_error=abs(baseline+back.sum()-risk)/100
    assert reconstruction_error<1e-12
    assert np.isclose(risk,100*curve['model'][pos,0],atol=1e-12,rtol=0)
    up=frame[pp>0].sort_values(['absolute_share_percent','term_id'],ascending=[False,True])
    down=frame[pp<0].sort_values(['absolute_share_percent','term_id'],ascending=[False,True])
    selected=pd.concat([up.head(8),down.head(6)]).copy()
    other=frame[~frame.term_id.isin(selected.term_id)]
    up_share=float(up.absolute_share_percent.sum());down_share=float(down.absolute_share_percent.sum())
    up_pp=float(up.termination_pp.sum());down_pp=float(down.termination_pp.sum())
    shown_share=float(selected.absolute_share_percent.sum())
    frame.to_csv(output/'all_contribution_shares_PRIVATE.csv',index=False)
    selected.to_csv(output/'overview_terms_PRIVATE.csv',index=False)
    group_rows=[]
    for order in [1,2]:
        for direction in ['升高','降低','零']:
            d=frame[(frame.order==order)&(frame.direction==direction)]
            group_rows.append(dict(order=order,direction=direction,terms=len(d),share_percent=float(d.absolute_share_percent.sum()),contribution_pp=float(d.termination_pp.sum())))
    pd.DataFrame(group_rows).to_csv(output/'contribution_groups_PRIVATE.csv',index=False)
    summary=dict(alias='匿名企业A',year=2024,arm='shallow_parent',horizon_calendar_months=24,
        risk_percent=risk,baseline_percent=baseline,total_absolute_contribution_pp=total,up_share_percent=up_share,down_share_percent=down_share,
        up_contribution_pp=up_pp,down_contribution_pp=down_pp,all_rows=len(frame),nonzero_rows=int((pp!=0).sum()),
        shown_rows=len(selected),shown_share_percent=shown_share,other_rows=len(other),other_share_percent=float(other.absolute_share_percent.sum()),
        denominator='sum(abs(all 83 main-input rows and 146 pair rows)); baseline excluded',
        reconstruction_error_probability=reconstruction_error,new_fits=0,source_case_unchanged=True,
        caveats=['local case attribution, not population feature importance','correlated audit encodings are not independent evidence',
                 'sign is relative to fixed product reference; not a causal intervention or a monotonic effect',
                 'share is not a probability increment or a calibration correction'])
    write(output/'share_summary_PRIVATE.json',summary)

    # Scientific image with a complete companion table; every share is computed
    # on all229 rows. Explicit overview selection does not renormalize the bars.
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
        'text.color':'#26384a','axes.labelcolor':'#26384a','xtick.color':'#526576','ytick.color':'#26384a',
        'savefig.facecolor':'white'})
    red='#c74b3d';teal='#197f78';muted='#66788a'
    fig=plt.figure(figsize=(16,11.6))
    fig.suptitle('匿名企业A｜升降方向＋解释贡献占比',fontsize=23,fontweight='bold',y=.975)
    fig.text(.5,.935,'现有2024年q2深度4＋收缩模型 · 24个月退市 · 同一企业、同一预测，未重新训练',ha='center',fontsize=12,color=muted)
    fig.text(.5,.900,f'参考基准 {baseline:.4f}% ＋ 升高项 {up_pp:.4f} pp − 降低项 {abs(down_pp):.4f} pp ＝ 模型预测 {risk:.4f}%',
        ha='center',fontsize=14,fontweight='bold')
    strip=fig.add_axes([.12,.830,.77,.035])
    strip.barh(0,up_share,color=red,height=.8);strip.barh(0,down_share,left=up_share,color=teal,height=.8)
    strip.text(up_share/2,0,f'升高风险项：{up_share:.2f}%',ha='center',va='center',color='white',fontweight='bold',fontsize=12)
    strip.text(up_share+down_share/2,0,f'降低风险项：{down_share:.2f}%',ha='center',va='center',color='white',fontweight='bold',fontsize=11)
    strip.set_xlim(0,100);strip.set_ylim(-.5,.5);strip.axis('off')
    fig.text(.5,.803,f'占比的共同分母＝全部主效应与交互的绝对贡献之和（{total:.4f}个百分点）；不含参考基准',ha='center',fontsize=11,color=muted)

    ax=fig.add_axes([.372,.235,.35,.51]);y=np.arange(len(selected),dtype=float);y[8:]+=.65
    values=selected.absolute_share_percent.to_numpy();colors=[red if v>0 else teal for v in selected.termination_pp]
    ax.barh(y,values,height=.58,color=colors)
    labels=[]
    for r in selected.itertuples():
        prefix='主效应' if r.order==1 else '交互'
        state=f' [{r.case_states}]' if r.order==1 and (r.case_states.startswith('=') or r.case_states=='缺失') else ''
        labels.append(f'{prefix}｜{r.label}{state}'+('*' if r.low_support else ''))
    ax.set_yticks(y,labels,fontsize=11);ax.set_ylim(y[-1]+.75,-.85)
    ax.set_xlim(0,max(values)*1.08);ax.set_xticks([0,3,6,9,12,15]);ax.set_xticklabels(['0%','3%','6%','9%','12%','15%'])
    ax.set_xlabel('解释贡献占比（条长表示贡献大小，颜色表示升降方向）',labelpad=12,fontsize=11)
    ax.grid(axis='x',color='#e7ebef');ax.set_axisbelow(True)
    ax.axhline(7.82,color='#cbd4dc',lw=.8,ls='--')
    ax.text(0,1.055,'主要升高项（前8项）／主要降低项（前6项）',transform=ax.transAxes,ha='left',fontsize=12,fontweight='bold')
    ax.text(1.10,1.055,'方向＋占比',transform=ax.transAxes,ha='center',fontsize=11,fontweight='bold')
    ax.text(1.43,1.055,'原始贡献\n（百分点）',transform=ax.transAxes,ha='center',fontsize=11,fontweight='bold')
    for yy,r,color in zip(y,selected.itertuples(),colors):
        direction='↑' if r.termination_pp>0 else '↓'
        ax.text(1.10,yy,f'{direction} {r.absolute_share_percent:.2f}%',transform=ax.get_yaxis_transform(),ha='center',va='center',color=color,fontsize=12,fontweight='bold')
        ax.text(1.43,yy,f'{r.termination_pp:+.4f}',transform=ax.get_yaxis_transform(),ha='center',va='center',color=color,fontsize=12)
    fig.text(.12,.153,f'本图14项占全部绝对贡献的 {shown_share:.2f}%；其余{len(other)}项占 {100-shown_share:.2f}%，完整CSV逐项保留。',fontsize=11)
    first=up.iloc[0]
    fig.text(.12,.122,f'例如“↑{first.absolute_share_percent:.2f}%”表示该项占全部绝对贡献的{first.absolute_share_percent:.2f}%，对应{first.termination_pp:+.4f}个百分点；占比不是概率增量。',fontsize=11)
    fig.text(.12,.091,'标记=0表示该状态不存在；三个审计标记存在编码关联，不能视为三份独立证据。交互单列，不重复分摊。',fontsize=10.5,color=muted)
    fig.text(.12,.061,'这是相对固定训练参考的模型归因，不是因果效应；负贡献不等于改善该指标的管理建议。',fontsize=10.5,color=muted)
    fig.text(.12,.031,'来源：原保存企业完整解释（83个主效应行＋146个交互行）。原始精度保留，比例换算后仍精确加总。',fontsize=10,color=muted)
    fig.savefig(output/'company_A_direction_share_PRIVATE.png',dpi=170);plt.close(fig)
    for name,h in sources.items():assert sha(Path(name))==h
    write(output/'execution_receipt.json',dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        new_fits=0,source_case_unchanged=True,all_rows=len(frame),all_shares_sum_percent=float(frame.absolute_share_percent.sum()),
        probability_reconstruction_error=reconstruction_error,script_sha256=sha(Path(__file__)),source_sha256=sources,
        renderer='matplotlib scientific static figure; full229-term companion exceeds standard bar category limits',
        privacy='anonymized company plot and contributions remain server-private; no Git publication of company detail',
        output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.receipt,a.reference,a.output)
