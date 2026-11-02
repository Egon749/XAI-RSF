"""Compare cohort min-max scores with the previously stated x3 illustration.

No fitted calibration or forest. Cohort scaling is display-only, not a frozen
prospective probability mapping. Every failed probability check is reported.
"""
from pathlib import Path
import argparse,copy,datetime,json,subprocess,time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crforest import CIFExpansion,CompetingRiskForest
from experiments.post_st_temporal_affine_v1.common import RUN as SOURCE,REPO,TIMES,YEARS,REFERENCE_SHA,load,arrays,sha,write,horizon_predictions,block_forest_fits
from experiments.post_st_temporal_affine_v1.analyze import point_outputs
from experiments.post_st_dropcolumn_selection_v1.common import metrics,evaluation_context
from experiments.post_st_ablation_v1.uncertainty import fast_metrics

PUB=REPO/'experiments/post_st_scaling_demo_v1'
ARMS=['baseline','minmax_score','times3']
NAMES={'baseline':'原预测','minmax_score':'归一化评分','times3':'固定3倍'}

def markdown(d):
    def f(x):
        if pd.isna(x):return 'NA'
        return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(d.columns)+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join(f(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'

def main(out):
    start=time.perf_counter();out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'summary').mkdir();block_forest_fits()
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    protocol=dict(kind='DESCRIPTIVE_SCALING_NOT_CALIBRATION',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),script_sha256=sha(Path(__file__)),
        source=str(SOURCE),arms=ARMS,fixed_multiplier=3.,multiplier_reason='fixed from preceding conversation example, no outcome-dependent search',
        normalization='Within each year and horizon: (p-min(p))/(max(p)-min(p)); display0..100. Uses whole evaluation cohort scores, not outcomes, hence batch-dependent, not prospective calibration.',
        outcome='first final termination1; recovery2 unchanged',years=YEARS,horizons=[12,24],unsupported='2025H24 no performance evaluation',
        clipping=False,new_forests=0,new_calibrators=0,adoption=False,privacy='Company predictions, maps and ranked figures remain private; only aggregates/code are public',
        metrics='AUC/AP/annual ceil20 capture; raw mean squared error for every numeric transform, explicitly not calibrated Brier for display scores/invalid competing CIF',
        preserve='Original models, CURRENT pointer and past predictions untouched')
    write(out/'protocol.json',protocol);(out/'run_snapshot.py').write_bytes(Path(__file__).read_bytes())
    paths,_,_=point_outputs();sources={};allrisk={};alllabels={};rows=[];counts=[];validation=[];constants=[]
    def read(p):sources[str(p)]=sha(p);return arrays(p)
    for year in YEARS:
        lab=read(SOURCE/f'evaluation_input/{year}_test_PRIVATE.npz');alllabels[year]=lab
        folder=paths[(year,24,'baseline')];z=read(folder/'eval_predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids'])
        tr=read(SOURCE/f'input/folds/{year}/train_PRIVATE.npz');ev=read(SOURCE/f'input/folds/{year}/eval_PRIVATE.npz')
        for p in [folder/'functions_PRIVATE.npz',folder/'fixed_forest_PRIVATE.json']:sources[str(p)]=sha(p)
        ex=CIFExpansion.load(folder/'functions_PRIVATE.npz');forest=CompetingRiskForest.load(folder/'fixed_forest_PRIVATE.json')
        scaled=copy.deepcopy(ex);scaled.mu=scaled.mu*np.array([3.,1.]);scaled.coefficients=scaled.coefficients*np.array([3.,1.])
        scaled.save(out/f'private/{year}_times3_FUNCTION_ONLY_NOT_VALIDATED_CIF.npz')
        reloaded=CIFExpansion.load(out/f'private/{year}_times3_FUNCTION_ONLY_NOT_VALIDATED_CIF.npz')
        X=np.vstack([tr['X'],ev['X']]);grid=np.unique(np.r_[0,ex.grid,TIMES]);error=0.;reload_error=0.;summax=0.;pmax=0.;badrows=0;mindiff=0.;components=[(j,) for j in ex.mains]+ex.pairs
        for first in range(0,len(X),32):
            xx=X[first:first+32];raw=forest.predict(xx,grid);expected=raw*np.array([3.,1.]);total=np.broadcast_to(scaled._coeff(grid)[1],expected.shape).copy()
            for key in components:total+=scaled.component(xx,key,grid)
            error=max(error,float(abs(total-expected).max()));reload_error=max(reload_error,float(abs(reloaded.predict(xx,grid)-expected).max()))
            summax=max(summax,float(expected.sum(-1).max()));pmax=max(pmax,float(expected[...,0].max()));badrows+=int((expected.sum(-1).max(1)>1+1e-10).sum());mindiff=min(mindiff,float(np.diff(expected,axis=1).min()))
        assert error<=1e-10 and reload_error<=1e-10 and mindiff>=-1e-12
        envelope=read(SOURCE/f'input/folds/{year}/envelope_PRIVATE.npz')['vertices'];positive=envelope[:,0]>0
        conservative_cap=float(np.min((1-envelope[positive,1])/envelope[positive,0]))
        validation.append(dict(year=year,train_eval_companies=len(X),native_time_checks=len(grid),full_component_error=error,reload_error=reload_error,
            scaled_curve_sum_max=summax,scaled_termination_max=pmax,companies_with_illegal_sum_somewhere=badrows,min_time_increment=mindiff,
            all_input_terminal_sum_upper_bound=float((envelope*np.array([3,1])).sum(1).max()),conservative_all_curve_multiplier_cap=conservative_cap,
            main_functions=len(ex.mains),pair_functions=len(ex.pairs),forest_fits=0))
        for h in [12,24]:
            endpoint=horizon_predictions(z['cif'],TIMES,lab[f'h{h}']);raw=endpoint[:,0];recovery=endpoint[:,1];low,high=float(raw.min()),float(raw.max());assert high>low
            constants.append(dict(year=year,months=h,minimum=low,maximum=high))
            transformed={'baseline':raw,'minmax_score':(raw-low)/(high-low),'times3':3*raw};y,w,known,supported,_=evaluation_context(lab,h,1)
            if supported:assert known.all() and np.all(w==1)
            order=np.lexsort((lab['sample_ids'],-raw));k=int(np.ceil(.2*len(raw)))
            for arm,p in transformed.items():
                np.testing.assert_array_equal(np.lexsort((lab['sample_ids'],-p)),order);m=metrics(lab,p,h,1)
                if supported:np.testing.assert_allclose(m['brier'],np.mean((p-y)**2),atol=1e-13,rtol=0)
                allrisk[(year,h,arm)]=p
                rows.append(dict(year=year,months=h,arm=arm,companies=len(p),supported=supported,observed=m['observed'],mean_value=float(p.mean()),min_value=float(p.min()),max_value=float(p.max()),
                    auc=m['auc'],ap=m['ap'],captured=m['captured'],events=m['target_events'] if supported else np.nan,checked=m['checked'],coverage=m['coverage'],
                    raw_squared_error=m['brier'],probability_brier=m['brier'] if arm=='baseline' else np.nan,
                    marginal_above_one=int((p>1+1e-10).sum()),joint_above_one_if_recovery_retained=int((p+recovery>1+1e-10).sum()),joint_sum_max=float((p+recovery).max()),
                    interpretation='CIF' if arm=='baseline' else 'display score, NOT probability' if arm=='minmax_score' else 'untuned multiplication, NOT certified competing CIF',reason=m['reason']))
            counts.append(dict(year=year,months=h,companies=len(raw),known_termination=int(y.sum()) if supported else np.nan,top20_companies=k,top20_captured=int(y[order[:k]].sum()) if supported else np.nan,
                all_three_rankings_identical=True,supported=supported))
            # Endpoint-only score functions remain exact, but are not CIF curves.
            timepoints=np.unique(lab[f'h{h}']);original_sum=np.broadcast_to(ex._coeff(timepoints)[1],(len(X),len(timepoints),2)).copy()
            for key in components:original_sum+=ex.component(X,key,timepoints)
            score=(original_sum[:,:,0]-low)/(high-low);expected=(forest.predict(X,timepoints)[:,:,0]-low)/(high-low)
            err=float(abs(score-expected).max());assert err<=1e-10
            validation.append(dict(year=year,months=h,score_endpoint_reconstruction_error=err,interpretation='score only; baseline shifted, all effects scaled; not a valid full CIF map'))
            pd.DataFrame(dict(sample_id=lab['sample_ids'][order],rank=np.arange(1,len(order)+1),original=raw[order],score_100=100*transformed['minmax_score'][order],times3=transformed['times3'][order],
                target=y[order] if supported else np.full(len(raw),np.nan),recovery=recovery[order])).to_csv(out/f'private/{year}_{h}_ranked_PRIVATE.csv',index=False)
    pooled=[]
    for h in [12,24]:
        years=YEARS if h==12 else [2023,2024];yy=np.concatenate([evaluation_context(alllabels[y],h,1)[0] for y in years]);ids=np.concatenate([alllabels[y]['sample_ids'] for y in years]);yr=np.concatenate([np.full(len(alllabels[y]['X']),y) for y in years])
        for a in ARMS:
            p=np.concatenate([allrisk[(y,h,a)] for y in years]);v=fast_metrics(yy,p,yr,ids)
            pooled.append(dict(months=h,arm=a,years=','.join(map(str,years)),companies=len(p),events=int(yy.sum()),auc=v[0],ap=v[1],raw_squared_error=v[2],mean_value=v[3],observed=float(yy.mean()),captured=v[6],coverage=v[7]))
        b=pooled[-3];t=pooled[-1]
        np.testing.assert_allclose([b[k] for k in ['auc','ap','captured']],[t[k] for k in ['auc','ap','captured']],atol=1e-13,rtol=0)
    np.savez_compressed(out/'private/transformed_values_PRIVATE.npz',**{'|'.join(map(str,k)):v for k,v in allrisk.items()})
    write(out/'private/normalization_constants_PRIVATE.json',constants)
    for name,data in [('COMPARISON',rows),('POOLED',pooled),('RANK_CHECKS',counts),('FUNCTION_PROBABILITY_CHECKS',validation)]:pd.DataFrame(data).to_csv(out/f'summary/{name}.csv',index=False)
    plot(out,allrisk,alllabels)
    for path,h in sources.items():assert sha(Path(path))==h
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    write(out/'source_hashes.json',sources)
    write(out/'execution_receipt.json',dict(status='COMPLETE_DESCRIPTIVE_DEMO',new_forests=0,new_calibrators=0,multiplier=3,selection_or_search=False,clipping=False,
        original_pointer_sha256=REFERENCE_SHA,original_pointer_unchanged=True,all_source_hashes_unchanged=True,seconds=time.perf_counter()-start,
        maximum_exact_function_error=max(v.get('full_component_error',v.get('score_endpoint_reconstruction_error',0)) for v in validation),
        probability_violations_reported_not_repaired=True,run=str(out),source_commit=protocol['source_commit'],script_sha256=protocol['script_sha256'],
        privacy='Ranks, company data, normalization constants and transformed function coefficients remain private'))
    publish(out,pd.DataFrame(rows),pd.DataFrame(pooled),pd.DataFrame(validation))
    print(pd.DataFrame(rows)[['year','months','arm','mean_value','max_value','auc','ap','raw_squared_error','joint_above_one_if_recovery_retained']].to_string(index=False));print(out)

def plot(out,risk,labels):
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42})
    y=2024;h=24;lab=labels[y];target=evaluation_context(lab,h,1)[0];order=np.lexsort((lab['sample_ids'],-risk[(y,h,'baseline')]));n=len(order);x=np.arange(1,n+1);k=int(np.ceil(.2*n))
    fig,axes=plt.subplots(1,3,figsize=(15,5.3),layout='constrained')
    for ax,a in zip(axes,ARMS):
        p=100*risk[(y,h,a)][order];hit=target[order]
        ax.axvspan(.5,k+.5,color='#167eaa',alpha=.08);ax.axvline(k+.5,color='#167eaa',ls='--',lw=1)
        ax.plot(x,p,color='#bbc5ce',lw=1);ax.scatter(x[~hit],p[~hit],color='#8cabb9',s=30);ax.scatter(x[hit],p[hit],color='#ca403b',s=40)
        ax.set_title(NAMES[a]);ax.set_xlim(.5,n+.5);ax.set_ylim(0,102 if a=='minmax_score' else max(5,float(p.max())*1.12));ax.set_xlabel('同一批企业的风险排名（高→低）')
        ax.set_ylabel('风险评分（分；不是概率）' if a=='minmax_score' else '原退市概率（%）' if a=='baseline' else '3×原概率（%；未校准）');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.text(.97,.95,f'平均 {p.mean():.2f}'+('分' if a=='minmax_score' else '%')+f'\n最高 {p.max():.2f}'+('分' if a=='minmax_score' else '%'),transform=ax.transAxes,ha='right',va='top',fontsize=10)
    fig.suptitle('2024年／24个月：同一45家公司，三种刻度；红点为先退市',fontsize=17)
    fig.supxlabel('三者AUC 0.770、AP 0.782；前20%均检查9家、捕获8/20例。只改刻度，不改变排名。',fontsize=11)
    for ext in ['png','pdf']:fig.savefig(out/f'private/SCALING_COMPARISON_2024_24_PRIVATE.{ext}',dpi=170)
    plt.close(fig)

def publish(out,m,p,v):
    import shutil
    for f in (out/'summary').iterdir():shutil.copy2(f,PUB/f.name)
    for name in ['protocol.json','execution_receipt.json']:shutil.copy2(out/name,PUB/name)
    annual=[]
    for y,h in [(2023,12),(2023,24),(2024,12),(2024,24),(2025,12)]:
        z=m[(m.year==y)&(m.months==h)].set_index('arm');b=z.loc['baseline'];s=z.loc['minmax_score'];t=z.loc['times3']
        annual.append({'年份':y,'月':h,'原均值%':100*b.mean_value,'归一化均分':100*s.mean_value,'3倍均值%':100*t.mean_value,'观察率%':100*b.observed,
            '原Brier':b.raw_squared_error,'3倍平方误差':t.raw_squared_error,'3倍后两原因和>1公司':int(t.joint_above_one_if_recovery_retained)})
    (PUB/'RESULTS_ZH.md').write_text('''# 归一化评分与固定3倍放大的实际对照

使用当前原83变量、q2、深度4＋父节点AJ10的保存预测，先平均三种子。没有森林重训、没有校准器拟合或倍率搜索。3倍来自此前对话的算术示例，在读取本轮误差前固定，不是从2023—2025结局挑选出的最优倍率。原模型与预测保持不变。

归一化按每个年度及期限的整批预测最小/最大值转换为0—100分，不使用结局决定尺度，但依赖这批评价公司的分数分布，属于批次展示，不是预先冻结的可部署概率校准。最高100分只表示该批次最高风险。固定3倍只放大退市预测，恢复预测不变；不进行任何裁剪。

'''+markdown(pd.DataFrame(annual))+'''

完整年度AUC/AP/Top20、原始值范围与概率冲突见COMPARISON.csv。三个方法在每年内部的排序完全一样，因此AUC/AP/Top20没有提高。相同3倍也保持跨年排序；各年分别归一化则不保证跨年排序。

原模型的Brier为合法概率预测的误差；3倍与归一化的raw_squared_error只是原样数值与0/1结局的平方误差。归一化评分不能直接称为概率；放大后若违反竞争风险和约束，也不能因为平方误差降低就宣布得到合法的竞争风险模型。为防止混淆，转换组的probability_brier列留NA，原样误差另列。2025年24个月随访不足，所有结局性能NA。

合并比较（12月2023—2025，24月仅2023/2024；Top20仍按年取名额后汇总）：

'''+markdown(p)+'''

完整函数与概率诊断见FUNCTION_PROBABILITY_CHECKS.csv。固定正倍率保留q2完整函数和时间单调性，但不保证退市加恢复小于等于1。归一化在固定期限可以将基线平移、各函数统一缩放后精确重构评分；不将这种端点评分冒充一条合法CIF曲线。原样记录全部概率冲突，不自动修改恢复风险或截断数值。

个人排序图、全部公司转换结果、实际归一化常数和派生函数保留服务器私有目录，公开目录只有汇总。仅为展示与机制核查，不更换参考模型，不据此选择新倍率或追加训练。原数据PROVISIONAL及已暴露年度的探索性限制延续。
''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
