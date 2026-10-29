"""Complete native explanation of the highest-risk captured 2024 company; no fit."""
from pathlib import Path
import argparse,datetime,json,math,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_prognosis_v1.common import arrays,sha,write,horizon_predictions
from .analyze import artifacts

LABELS={
'gross_margin':'毛利率','operating_margin':'营业利润率','net_margin':'净利率','roa':'ROA',
'operating_profit_to_assets':'营业利润/总资产','net_profit_to_assets':'净利润/总资产',
'financial_expense_to_revenue':'财务费用/营业收入','loss_flag':'亏损标记',
'asset_impairment_loss_to_assets':'资产减值损失/总资产','current_ratio':'流动比率','quick_ratio':'速动比率',
'cash_ratio':'现金比率','asset_liability_ratio':'资产负债率','net_debt_to_assets':'净债务/总资产',
'cash_to_assets':'现金/总资产','short_debt_to_assets':'短期债务/总资产','long_debt_to_assets':'长期债务/总资产',
'log_cash_to_short_debt':'现金/短债（对数）','interest_coverage':'利息保障倍数','cfo_to_assets':'经营现金流/总资产',
'cfo_to_current_liabilities':'经营现金流/流动负债','cfo_to_net_income':'经营现金流/净利润',
'negative_equity_flag':'净资产为负标记','negative_cfo_flag':'经营现金流为负标记',
'ar_to_current_assets':'应收账款/流动资产','inventory_to_current_assets':'存货/流动资产',
'working_capital_to_assets':'营运资本/总资产','total_asset_turnover':'总资产周转率','ar_turnover':'应收账款周转率',
'inventory_turnover':'存货周转率','cfo_to_revenue':'经营现金流/营业收入','accrual_intensity':'应计强度',
'ap_to_assets':'应付账款/总资产','noncurrent_assets_to_assets':'非流动资产/总资产','fixed_assets_to_assets':'固定资产/总资产',
'revenue_growth':'营业收入增长','total_assets_growth':'总资产增长','operating_profit_growth':'营业利润增长',
'net_profit_growth':'净利润增长','profit_total_growth':'利润总额增长','cfo_growth':'经营现金流增长',
'investing_cf_growth':'投资现金流增长','financing_cf_growth':'筹资现金流增长','gross_margin_change':'毛利率变化',
'leverage_change':'杠杆变化','investing_cf_to_assets':'投资现金流/总资产','financing_cf_to_assets':'筹资现金流/总资产',
'ln_total_assets':'总资产（对数）','log_price_last':'最近股价（对数）','log_total_mktcap_last':'总市值（对数）',
'log_float_mktcap_last':'流通市值（对数）','ret_cum_1m':'近1个月累计收益率','ret_vol_1m':'近1个月收益波动率',
'max_drawdown_1m':'近1个月最大回撤','log_amihud_illiq_1m':'近1个月非流动性（对数）',
'ret_cum_3m':'近3个月累计收益率','ret_vol_3m':'近3个月收益波动率','max_drawdown_3m':'近3个月最大回撤',
'total_mktcap_change_3m':'近3个月总市值变化','log_amihud_illiq_3m':'近3个月非流动性（对数）',
'downside_vol_3m':'近3个月下行波动率','var_5pct_3m':'近3个月收益5%分位','es_5pct_3m':'近3个月尾部平均收益',
'extreme_negative_days_3m':'近3个月极端下跌天数','turnover_mean_3m':'近3个月平均换手率',
'log_amount_mean_3m':'近3个月成交额均值（对数）','intraday_range_mean_3m':'近3个月平均日内振幅',
'zero_amount_ratio_3m':'近3个月零成交额占比','ret_cum_12m':'近12个月累计收益率','ret_vol_12m':'近12个月收益波动率',
'max_drawdown_12m':'近12个月最大回撤','total_mktcap_change_12m':'近12个月总市值变化',
'fin_period_age_scaled':'财报期末陈旧度（缩放）','fin_publication_age_scaled':'财报披露陈旧度（缩放）',
'market_missing_coverage':'市场数据缺失覆盖','entry_is_star_st':'入组时为*ST标记',
'prior_audit_standard':'标准无保留意见标记','prior_audit_qualified':'保留意见标记',
'prior_audit_adverse':'否定意见标记','prior_audit_disclaimer':'无法表示意见标记',
'prior_audit_explanatory':'解释性无保留意见标记','prior_audit_qualified_matter':'保留意见加事项段标记',
'prior_audit_going_concern':'持续经营疑虑标记'}
RED='#CC4E3D';TEAL='#23877C';BLUE='#315F94';GRAY='#9AA4AD'


def run(a):
    root=a.run;out=a.output;assert out.is_dir() and not any(out.iterdir())
    sources={}
    def source(p):
        p=Path(p);sources[str(p)]=sha(p);return p
    plan=json.loads(source(root/'input/folds/2024/plan.json').read_text());names=plan['feature_names']
    assert len(names)==83 and set(names)<=set(LABELS)
    tr=arrays(source(root/'input/folds/2024/train_PRIVATE.npz'));ev=arrays(source(root/'evaluation_input/2024_test_PRIVATE.npz'))
    meta=pd.read_parquet(source(root/'evaluation_input/2024_test_metadata_PRIVATE.parquet'))
    np.testing.assert_array_equal(meta.sample_id,ev['sample_ids']);assert ev['label_eligible'].all()
    folders={str(s):p for y,arm,s,p in artifacts(root) if y==2024 and arm=='shallow_parent'}
    predictions=[];forests=[]
    for seed in ['42','101','202']:
        folder=folders[seed];hh=json.loads(source(folder/'hashes.json').read_text())
        for name in ['forest_PRIVATE.json','predictions_PRIVATE.npz','functions_PRIVATE.npz']:
            assert sha(source(folder/name))==hh[name]
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json')
        assert f.config.max_depth==4 and f.config.path_budget==2 and f.config.n_trees==32 and f.config.seed==int(seed)
        assert all(n.get('leaf_shrinkage',{}).get('strength')==10 for tree in f.trees_ for n in tree if 'cif' in n)
        forests.append(f);p=arrays(folder/'predictions_PRIVATE.npz')
        np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids'])
        predictions.append(horizon_predictions(p['cif'],p['times'],ev['h24']))
    avg=np.mean(predictions,axis=0);order=np.lexsort((ev['sample_ids'],-avg[:,0]));top=int(np.ceil(len(order)*.2))
    eligible=[int(i) for i in order[:top] if ev['event'][i]==1 and ev['analysis_time'][i]<=24]
    assert len(eligible)==8;index=eligible[0];rank=int(np.flatnonzero(order==index)[0]+1);row=meta.iloc[index]
    X=ev['X'][index:index+1];h=int(ev['h24'][index])
    expansion_file=source(root/'ensemble_private/2024_shallow_parent_PRIVATE.npz')
    cc=pd.read_csv(source(root/'evaluation/ensemble_complexity.csv'))
    assert sha(expansion_file)==cc[(cc.test_year==2024)&(cc.arm=='shallow_parent')].iloc[0].functions_sha256
    ex=CIFExpansion.load(expansion_file);assert ex.reference_n==len(tr['X'])==345
    keys=[(j,) for j in range(len(names))]+ex.pairs
    times=np.unique(np.r_[0.,ex.grid,365.,366.,730.,731.]);position=int(np.flatnonzero(times==h)[0])
    values=np.stack([ex.component(X,key,times)[0] for key in keys])
    baseline=ex._coeff(times)[1];summed=baseline+values.sum(axis=0)
    model_curve=np.mean([f.predict(X,times)[0] for f in forests],axis=0)
    errors=dict(full_native_component_sum=float(abs(summed-model_curve).max()),
                exported_evaluator=float(abs(ex.predict(X,times)[0]-model_curve).max()),
                saved_prediction=float(abs(model_curve[position]-avg[index]).max()))
    assert max(errors.values())<1e-10
    assert model_curve.min()>=0 and model_curve.sum(-1).max()<=1+1e-12 and np.diff(model_curve,axis=0).min()>=-1e-12
    # Re-load the complete exported table without growing or changing any tree.
    errors['reload']=float(abs(CIFExpansion.load(expansion_file).predict(X,times)[0]-model_curve).max())
    descriptors=[];states={}
    for j,name in enumerate(names):
        v=X[0,j];ref=tr['X'][:,j];finite=ref[np.isfinite(ref)]
        percentile=float(100*((finite<v).sum()+.5*(finite==v).sum())/len(finite)) if np.isfinite(v) and len(finite) else None
        binary=name.endswith('_flag') or name.startswith('prior_audit_') or name=='entry_is_star_st'
        state='缺失' if np.isnan(v) else (f'={int(v)}' if binary else f'P{percentile:.0f}')
        states[j]=state
        descriptors.append(dict(index=j,name=name,label=LABELS[name],raw_model_input=None if np.isnan(v) else float(v),
            state=state,training_nonmissing_midrank_percentile=percentile,main_used=j in ex.mains))
    rows=[]
    for i,key in enumerate(keys):
        mask=np.ones(len(tr['X']),bool)
        for j in key:
            cuts=ex.cuts(key,j);train_bins=np.searchsorted(cuts,tr['X'][:,j],side='left')
            train_bins[np.isnan(tr['X'][:,j])]=len(cuts)+1
            case_bin=len(cuts)+1 if np.isnan(X[0,j]) else int(np.searchsorted(cuts,X[0,j],side='left'))
            mask &= train_bins==case_bin
        support=int(mask.sum());used=key[0] in ex.mains if len(key)==1 else True
        rows.append(dict(term_id=f'M{key[0]+1:02d}' if len(key)==1 else f'I{ex.pairs.index(key)+1:03d}',
            order=len(key),features=','.join(map(str,key)),names=' × '.join(names[j] for j in key),
            label=' × '.join(LABELS[names[j]] for j in key),
            case_states=' / '.join(states[j] for j in key),used=used,
            termination_contribution=float(values[i,position,0]),termination_pp=float(values[i,position,0]*100),
            recovery_contribution=float(values[i,position,1]),joint_training_cell_support=support,
            low_support=bool(used and support<10)))
    frame=pd.DataFrame(rows);main=frame[frame.order==1];pairs=frame[frame.order==2]
    base=float(baseline[position,0])*100;risk=float(model_curve[position,0])*100
    main_up=float(main.loc[main.termination_pp>0,'termination_pp'].sum());main_down=float(main.loc[main.termination_pp<0,'termination_pp'].sum())
    pair_up=float(pairs.loc[pairs.termination_pp>0,'termination_pp'].sum());pair_down=float(pairs.loc[pairs.termination_pp<0,'termination_pp'].sum())
    np.testing.assert_allclose(base+main_up+main_down+pair_up+pair_down,risk,atol=1e-10,rtol=0)
    frame.to_csv(out/'all_contributions_PRIVATE.csv',index=False);pd.DataFrame(descriptors).to_csv(out/'all_input_states_PRIVATE.csv',index=False)
    np.savez_compressed(out/'full_curve_reconstruction_PRIVATE.npz',times=times,baseline=baseline,terms=values,model=model_curve,component_sum=summed)
    info=dict(alias='匿名企业A',selection='highest predicted risk among top20% actually first-terminated companies; no explanation-based selection',
        private_sample_id=str(row.sample_id),private_company_group=str(row.group),private_company_code=str(row.code),rank=rank,test_companies=45,
        top20_checked=top,top20_captured=len(eligible),origin=str(row.origin.date()),observed_termination=str(row.stop.date()),event_days=int(row.time),
        horizon_calendar_months=24,horizon_days=h,base_percent=base,predicted_percent=risk,
        main_up_pp=main_up,main_down_pp=main_down,main_net_pp=main_up+main_down,
        interaction_up_pp=pair_up,interaction_down_pp=pair_down,interaction_net_pp=pair_up+pair_down,
        actual_main_functions=len(ex.mains),actual_pair_functions=len(ex.pairs),unused_inputs=len(names)-len(ex.mains),
        displayed_rows=len(frame),low_support_pair_cells=int(pairs.low_support.sum()),
        zero_support_pair_cells=int((pairs.joint_training_cell_support==0).sum()),
        errors=errors,reference='product of empirical feature marginals of the frozen 345-company training reference; not observed event rate or mean joint-sample prediction')
    write(out/'case_summary_PRIVATE.json',info)
    # TrueType embeds consistently in both the scientific PNG and PDF exports.
    font='/home/oliver/.local/share/fonts/microsoft/simhei.ttf';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':10,
        'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A0AAB4',
        'text.color':'#243544','axes.labelcolor':'#243544','xtick.color':'#566775','ytick.color':'#344856',
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    def signed(v,digits=4):
        return f'{0. if abs(v)<.5*10**(-digits) else v:+.{digits}f}'
    def small_bars(ax,subset,title):
        pos=subset[subset.termination_pp>0].nlargest(5,'termination_pp')
        neg=subset[subset.termination_pp<0].nsmallest(3,'termination_pp')
        selected=pd.concat([pos,neg]).copy();rest=subset[~subset.term_id.isin(selected.term_id)]
        selected['chart_label']=selected.apply(lambda r:r.label+' '+r.case_states if r.order==1 else r.label,axis=1)
        names_here=selected.chart_label.tolist()+[f'其余{len(rest)}项合计（全量图逐项展开）']
        vals=np.r_[selected.termination_pp.to_numpy(),rest.termination_pp.sum()]
        ys=np.arange(len(vals));colors=[RED if x>=0 else TEAL for x in vals]
        ax.barh(ys,vals,color=colors,height=.63)
        lim=max(abs(vals).max(),.01);ax.set_xlim(min(vals.min(),0)-lim*.22,max(vals.max(),0)+lim*.35)
        for y,v in zip(ys,vals):ax.annotate(signed(v,3),(v,y),xytext=(4 if v>=0 else -4,0),textcoords='offset points',
            ha='left' if v>=0 else 'right',va='center',fontsize=9)
        ax.set_yticks(ys,names_here,fontsize=9);ax.invert_yaxis();ax.axvline(0,color='#8896A3',lw=.8)
        ax.set_xlabel('对退市CIF的贡献（百分点）');ax.set_title(title,loc='left',fontsize=12,pad=12)
        ax.grid(axis='x',color='#E8ECF0');ax.set_axisbelow(True)
    with PdfPages(out/'company_A_complete_explanation_PRIVATE.pdf') as pdf:
        fig=plt.figure(figsize=(17,12))
        fig.suptitle('匿名企业A｜2024年「深度4＋收缩」完整风险解释',fontsize=21,y=.975,fontweight='bold')
        fig.text(.5,.935,f'预测风险 {risk:.2f}%　｜　排名 {rank}/45，位于前20%　｜　起点后 {int(row.time)} 天实际先退市',ha='center',fontsize=13,color='#4F6373')
        ax=fig.add_axes([.08,.57,.87,.29])
        labels=['参考基准','主效应：升风险','主效应：降风险','交互：升风险','交互：降风险','最终预测']
        changes=[base,main_up,main_down,pair_up,pair_down,risk];running=0.
        for i,value in enumerate(changes):
            if i in [0,5]:start=0.;end=value;color=GRAY if i==0 else BLUE
            else:start=running;end=start+value;color=RED if value>=0 else TEAL
            ax.bar(i,abs(end-start),bottom=min(start,end),width=.62,color=color,zorder=3)
            if i<5:ax.plot([i+.31,i+.69],[end,end],color='#9EAAB5',lw=.9)
            label=f'{value:.4f}%' if i in [0,5] else signed(value)+' pp'
            ax.text(i,max(start,end)+.45,label,ha='center',fontsize=11,fontweight='bold')
            running=end
        ax.set_xticks(range(6),labels,fontsize=11);ax.set_ylim(0,24);ax.set_ylabel('退市概率（%）')
        ax.grid(axis='y',color='#E4EAF0');ax.set_axisbelow(True)
        ax.set_title('完整加总：所有主效应与交互均纳入，没有隐藏残差',loc='left',fontsize=12,pad=17)
        fig.text(.5,.515,f'{base:.4f}% ＋ 主效应净贡献 {main_up+main_down:.4f} pp ＋ 交互净贡献 {pair_up+pair_down:.4f} pp ≈ {risk:.4f}%',ha='center',fontsize=13,color=BLUE)
        left=fig.add_axes([.205,.16,.235,.285]);right=fig.add_axes([.745,.16,.225,.285])
        small_bars(left,main[main.used],'主要升降项：主效应（其余项精确合计）')
        small_bars(right,pairs,'主要升降项：二元交互（其余项精确合计）')
        fig.text(.5,.083,f'全量附图：65个入模主效应＋146个交互；另列18个未使用输入的零贡献。最大全曲线重构误差 {max(errors.values()):.2e}。',ha='center',fontsize=10)
        fig.text(.5,.052,'红色提高、绿色降低参考风险；pp为百分点。P为该输入在训练非缺失样本中的百分位，0/1标记保留其实际状态。',ha='center',fontsize=10)
        fig.text(.5,.024,'贡献基于固定训练参考分解，非因果效应；审计标记存在关联，不能视作多份独立证据。只读取现有模型，未训练。',ha='center',fontsize=10,color='#5C6C79')
        fig.savefig(out/'company_A_overview_PRIVATE.png',dpi=180);pdf.savefig(fig);plt.close(fig)
        def full_page(subset,title,filename,pair=False):
            chunks=[subset.iloc[ix] for ix in np.array_split(np.arange(len(subset)),2)]
            fig,axes=plt.subplots(1,2,figsize=(18,14))
            fig.subplots_adjust(left=.235,right=.96,bottom=.11,top=.87,wspace=1.45)
            fig.suptitle(title,fontsize=19,y=.973,fontweight='bold')
            fig.text(.5,.94,f'匿名企业A｜24个月退市CIF {risk:.4f}%｜每项单独列出，原精度见CSV；原精度加总可精确重构',ha='center',fontsize=11)
            for ax,part in zip(axes,chunks):
                y=np.arange(len(part));vv=part.termination_pp.to_numpy()
                labels_here=[]
                for rr in part.itertuples():
                    label=rr.term_id+'  '+rr.label
                    if not pair:label+=' '+rr.case_states+('〔未入模〕' if not rr.used else '')
                    labels_here.append(label)
                ax.barh(y,vv,color=[RED if v>0 else TEAL if v<0 else GRAY for v in vv],height=.65)
                ax.scatter(np.zeros((vv==0).sum()),y[vv==0],color=GRAY,s=9,zorder=4)
                limit=max(abs(subset.termination_pp).max(),.001)
                ax.set_xlim(min(0.,subset.termination_pp.min())-limit*.22,max(0.,subset.termination_pp.max())+limit*.48)
                for yy,v in zip(y,vv):ax.annotate(signed(v),(v,yy),xytext=(3 if v>=0 else -3,0),textcoords='offset points',
                    va='center',ha='left' if v>=0 else 'right',fontsize=8)
                ax.set_yticks(y,labels_here,fontsize=8.6);ax.invert_yaxis();ax.axvline(0,color='#9CA7B1',lw=.7)
                ax.grid(axis='x',color='#E8ECF0');ax.set_axisbelow(True);ax.set_xlabel('CIF贡献（百分点）')
                ax.text(1.01,1.015,'训练落格\n公司数',transform=ax.transAxes,ha='center',va='bottom',fontsize=8)
                for yy,rr in zip(y,part.itertuples()):
                    ax.text(1.015,yy,str(rr.joint_training_cell_support)+('*' if rr.low_support else ''),
                        transform=ax.get_yaxis_transform(),va='center',ha='left',fontsize=8,color=RED if rr.low_support else '#5B6B77')
            fig.text(.5,.065,'按贡献绝对值排序；显示四位小数，原精度见CSV。星号*表示该函数所用分区内训练参考公司少于10家，不等于目标事件支持。',ha='center',fontsize=10)
            fig.text(.5,.035,'缺失状态单独保留；不删除零贡献项，不把交互分摊给主效应，不加入近似解释或裁剪残差。',ha='center',fontsize=10,color='#5B6B77')
            fig.savefig(out/filename,dpi=170);pdf.savefig(fig);plt.close(fig)
        ordered_main=main.assign(mag=main.termination_pp.abs()).sort_values(['used','mag','term_id'],ascending=[False,False,True])
        full_page(ordered_main,'全部主效应｜83个输入：65个入模＋18个未使用','company_A_all_main_effects_PRIVATE.png')
        ordered_pairs=pairs.assign(mag=pairs.termination_pp.abs()).sort_values(['mag','term_id'],ascending=[False,True])
        for i,ix in enumerate(np.array_split(np.arange(len(ordered_pairs)),2)):
            part=ordered_pairs.iloc[ix]
            full_page(part,f'全部二元交互｜第{i+1}/2页（共146项，每项均列出）',f'company_A_all_interactions_{i+1}_PRIVATE.png',True)
    summary=f'''# 匿名企业A：现有2024深度4＋收缩模型的完整解释

选择规则：在前20%且实际先退市的企业中，取预测风险排名最高者。排名{rank}/45，预测{risk:.8f}%，起点后{int(row.time)}天实际先退市。仅读取已保存的三个32树模型及96树完整函数表，没有重新训练或选择解释更好看的案例。

完整加总：基准{base:.12f}%＋主效应净贡献{main_up+main_down:.12f}个百分点＋交互净贡献{pair_up+pair_down:.12f}个百分点＝{risk:.12f}%。
主效应正项{main_up:.12f}、负项{main_down:.12f}；交互正项{pair_up:.12f}、负项{pair_down:.12f}。
模型65个主效应、146个二元交互全部纳入；总表还列出18个未使用输入的0贡献，合计229行。

总览图中的其余项是原精度带符号求和，不是拟合残差；PDF后续三页完整列出全部项及其训练参考分区支持数。83个输入的原始值、缺失状态与训练百分位单独留在CSV，真实公司对应关系仅在私有摘要。

这些贡献是对24个月退市CIF的百分点贡献，不是对数风险率或风险率乘数，也不是因果效应。符号表示相对固定参考分解的方向，不等于改变该变量就会产生同量变化。基准使用345家公司各输入边际分布的乘积，不能解释为实际退市率或训练预测的简单平均；不同中心化参考可以改变归因。

审计意见标记相互关联，标准无保留意见标记=0与无法表示意见标记=1不是两份独立审计证据。主效应和交互均保留，以便检查这种编码关系。交互落格支持少于10家的有{int(pairs.low_support.sum())}项，零支持有{int((pairs.joint_training_cell_support==0).sum())}项；精确解释不证明这些局部函数可靠、稳定或因果可辨识。

完整模型直接预测、保存预测、导出函数和逐项加总在所有原生时间、两个原因核验，最大差{max(errors.values()):.17g}。没有人工裁剪或补残差。以上均属于既有探索性样本，不能用这一个命中案例替代整体评价。
'''
    (out/'CASE_EXPLANATION_ZH.md').write_text(summary)
    for path,hsh in sources.items():assert sha(Path(path))==hsh
    write(out/'execution_receipt.json',dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        new_fits=0,year=2024,arm='shallow_parent',seeds=[42,101,202],trees=96,case_selection='highest risk in top20% with observed first termination',
        feature_inputs=83,actual_main_functions=65,actual_pair_functions=146,all_component_rows=229,all_components_retained=True,
        full_native_times_checked=len(times),causes_checked=2,errors=errors,source_sha256=sources,script_sha256=sha(Path(__file__)),
        renderer='matplotlib scientific PNG/PDF; full signed contribution panels exceed standard bar category/height limits',
        privacy='all company identity, values, predictions, figures and contributions remain server-private',
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in info.items() if not k.startswith('private_')},indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
