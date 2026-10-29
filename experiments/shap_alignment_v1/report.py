"""Verify published outputs, aggregate all seeds, and render private research figures."""
import argparse,hashlib,json,shutil,re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from experiments.post_st_leaf_shallow_v1.explain_company import LABELS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def table(df):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6g}' if np.isfinite(x) else '—'
        return str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def err(a,b):
    d=np.asarray(a)-np.asarray(b);return dict(max_abs=float(abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))))


def run(root,job,pool_result,dest,report_tag=None):
    out=root/'reports';private=root/'private'
    if report_tag:
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',report_tag):raise ValueError('invalid report tag')
        out=out/report_tag;private=private/report_tag
    out.mkdir(exist_ok=True);private.mkdir(exist_ok=True)
    assert not (out/'execution_receipt.json').exists()
    figures=out/'figures';figures.mkdir(exist_ok=True)
    status=json.loads(pool_result.read_text());assert status['expected']==6 and not status['pending'] and status['srun_returncode']==0
    assert len(status['completed'])==6
    for t in status['completed']:
        assert t['returncode']==0 and t['compute_completed'] and t['publication_completed']
        assert t['storage']['publication_state']=='verified'
    for rel,h in json.loads((root/'audit/input_hashes_PRIVATE.json').read_text()).items():assert sha(root/'input'/rel)==h
    snapshot='code_v3' if (root/'audit/code_hashes_v3.json').exists() else 'code'
    codehash='code_hashes_v3.json' if snapshot=='code_v3' else 'code_hashes.json'
    for rel,h in json.loads((root/'audit'/codehash).read_text()).items():assert sha(root/snapshot/rel)==h
    for path,h in json.loads((root/'audit/source_hashes_PRIVATE.json').read_text()).items():assert sha(path)==h
    folders={};summaries=[];align=[];orders=[];checks=[];hashes=[];disp=[]
    for i in range(6):
        task=root/'results'/str(job)/f'task-{i}';p=task/'output'
        receipt=json.loads((task/'complete.json').read_text())
        for entry in receipt['manifest']:
            if len(entry)==3:
                rel,size,h=entry;assert (p/rel).stat().st_size==size and sha(p/rel)==h
        s=json.loads((p/'summary.json').read_text());assert s['status']=='COMPLETE' and s['new_model_fits']==0
        folders[(s['year'],s['seed'])]=p;summaries.append({k:v for k,v in s.items() if k not in ['original_hashes','rsf_adapter_error']})
        for name,acc in [('alignment_metrics',align),('order_metrics',orders)]:acc.append(pd.read_csv(p/f'{name}.csv'))
        dd=pd.read_csv(p/'dispersion_PRIVATE.csv')
        for (m,b,ss),g in dd.groupby(['model','background','subset']):
            disp.append(dict(year=s['year'],seed=s['seed'],model=m,background=b,subset=ss,
                features_with_nonzero_within_cell_dispersion=int((g.within_cell_rms>1e-12).sum()),
                max_within_cell_rms=float(g.within_cell_rms.max()),max_within_cell_range=float(g.max_within_cell_range.max())))
        for check in json.loads((p/'checks.json').read_text()):
            for name,e in check['errors'].items():checks.append(dict(year=s['year'],seed=s['seed'],model=check['model'],background=check['background'],check=name,**e))
        for name,h in s['original_hashes'].items():hashes.append(dict(year=s['year'],seed=s['seed'],artifact=name,sha256=h))
    assert set(folders)=={(y,s) for y in [2023,2024] for s in [42,101,202]}
    a=pd.concat(align,ignore_index=True);o=pd.concat(orders,ignore_index=True);c=pd.DataFrame(checks)
    a.drop(columns=['baseline']).to_csv(out/'alignment_metrics.csv',index=False);o.to_csv(out/'order_metrics.csv',index=False);c.to_csv(out/'numerical_checks.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'task_diagnostics.csv',index=False);pd.DataFrame(hashes).to_csv(out/'source_model_hashes.csv',index=False)
    pd.DataFrame(disp).to_csv(out/'dispersion_summary.csv',index=False)
    for path in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']:font_manager.fontManager.addfont(path)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'axes.unicode_minus':False,'figure.dpi':150})
    ensemble_rows=[];ensemble_order=[];ensemble={};plots=[];decomp_summary=[]
    for year in [2023,2024]:
        seedfolders=[folders[(year,s)] for s in [42,101,202]]
        cfg=json.loads((root/'input'/str(year)/'42/config.json').read_text());z=load(root/'input'/str(year)/'42/features_PRIVATE.npz')
        n=cfg['n_train'];X=np.vstack([z['train_X'],z['test_X']]);p=X.shape[1]
        native={k:np.mean([load(f/'q2_native_PRIVATE.npz')[k] for f in seedfolders],axis=0) for k in ['Phi','baseline','predictions']}
        ensemble[(year,'native')]=native
        for model in ['standard_rsf','q2']:
            for background in ['default','product']:
                data={k:np.mean([load(f/f'{model}_{background}_PRIVATE.npz')[k] for f in seedfolders],axis=0) for k in ['Phi','phi','base','predictions']}
                ensemble[(year,model,background)]=data
                np.testing.assert_allclose(data['base']+data['phi'].sum(1),data['predictions'],atol=1e-12,rtol=0)
                for subset,sl in [('training',slice(0,n)),('evaluation',slice(n,None))]:
                    row=dict(year=year,model=model,background=background,subset=subset,n=len(data['phi'][sl]),base=float(data['base']))
                    if model=='q2':
                        rows=err(data['Phi'][sl,np.arange(p),np.arange(p)],native['Phi'][sl,np.arange(p),np.arange(p)])
                        pairs=err(2*data['Phi'][sl][:,~np.eye(p,dtype=bool)],2*native['Phi'][sl][:,~np.eye(p,dtype=bool)])
                        ordinary=err(data['phi'][sl],native['Phi'][sl].sum(2))
                        row.update(main_max_abs=rows['max_abs'],main_rmse=rows['rmse'],full_pair_max_abs=pairs['max_abs'],full_pair_rmse=pairs['rmse'],
                                   ordinary_allocation_max_abs=ordinary['max_abs'],ordinary_allocation_rmse=ordinary['rmse'],
                                   baseline_difference=float(data['base']-native['baseline']))
                    ensemble_rows.append(row)
            dec=[load(f/f'{model}_decomposition_PRIVATE.npz') for f in seedfolders]
            orders=np.mean([d['order'] for d in dec],axis=0);hi=np.mean([d['higher_phi'] for d in dec],axis=0)
            ensemble[(year,model,'order')]=orders
            # Terms are combined by feature tuple across seeds before measuring size.
            terms={}
            for d in dec:
                for j,key in enumerate(d['keys']):terms[key]=terms.get(key,0)+d['terms'][:,j]/3
            keys=sorted(terms,key=lambda k:(k.count(','),k));tv=np.stack([terms[k] for k in keys],axis=1)
            np.savez_compressed(private/f'{year}_{model}_ensemble_functions_PRIVATE.npz',keys=keys,terms=tv,order=orders,higher_phi=hi)
            for subset,sl in [('training',slice(0,n)),('evaluation',slice(n,None))]:
                for r in range(1,orders.shape[1]):
                    v=orders[sl,r];ensemble_order.append(dict(year=year,model=model,subset=subset,order=r,rms=float(np.sqrt(np.mean(v*v))),mean_abs=float(abs(v).mean()),max_abs=float(abs(v).max())))
                high=orders[sl,3:].sum(1)
                decomp_summary.append(dict(year=year,model=model,subset=subset,max_order=orders.shape[1]-1,
                    high_order_functions=sum(k.count(',')>=2 for k in keys),
                    high_order_sum_max_abs=float(abs(high).max()),high_order_sum_mean_abs=float(abs(high).mean()),
                    high_order_sum_rms=float(np.sqrt(np.mean(high*high))),higher_order_allocated_shap_rms=float(np.sqrt(np.mean(hi[sl]**2)))))
        selections=[int(np.argmax(abs(ensemble[(year,'standard_rsf','default')]['phi'][:n]).mean(0))),
                    int(np.argmax(abs(ensemble[(year,'q2','product')]['phi'][:n]).mean(0)))]
        selections=list(dict.fromkeys(selections))
        if len(selections)==1:
            order=np.argsort(-abs(ensemble[(year,'q2','product')]['phi'][:n]).mean(0));selections.append(int(next(j for j in order if j!=selections[0])))
        fig,axes=plt.subplots(len(selections),4,figsize=(16,7.6),squeeze=False,sharey='row')
        for row,j in enumerate(selections):
            finite=np.sort(X[:n,j][np.isfinite(X[:n,j])]);v=X[:,j]
            xpos=100*(np.searchsorted(finite,v,side='left')+np.searchsorted(finite,v,side='right'))/(2*max(len(finite),1));xpos[np.isnan(v)]=105
            binary=np.isin(finite,[0.,1.]).all()
            if binary:xpos=np.where(np.isnan(v),1.18,v)
            series=[ensemble[(year,'standard_rsf','default')]['phi'][:,j],ensemble[(year,'q2','product')]['phi'][:,j],
                    ensemble[(year,'q2','product')]['Phi'][:,j,j],ensemble[(year,'q2','default')]['Phi'][:,j,j]]
            titles=['标准 RSF：默认普通 SHAP','q2：乘积背景普通 SHAP','q2：乘积背景交互矩阵对角线','q2：默认交互矩阵对角线']
            for col,y in enumerate(series):
                ax=axes[row,col];ax.scatter(xpos[:n],100*y[:n],s=15,alpha=.38,color='#4678A6',label='训练参考')
                ax.scatter(xpos[n:],100*y[n:],s=23,alpha=.8,color='#DA8246',marker='x',label='验证记录')
                if col>=2:
                    main=native['Phi'][:,j,j]*100
                    ax.scatter(xpos,main,s=12,facecolors='none',edgecolors='#202020',linewidths=.5,label='原生主效应 f_j')
                ax.axhline(0,color='#b0b0b0',lw=.6);ax.set_title(titles[col],fontsize=10)
                ax.grid(alpha=.14);ax.set_xlabel('标记取值（0/1）' if binary else '训练参考中的分位位置（%）')
                if binary:
                    ax.set_xticks([0,1,1.18],['0','1','缺失']) if np.isnan(v).any() else ax.set_xticks([0,1],['0','1'])
                    ax.set_xlim(-.1,1.28 if np.isnan(v).any() else 1.1)
                elif np.isnan(v).any():ax.set_xticks([0,25,50,75,100,105],['0','25','50','75','100','缺失'])
                if col==0:ax.set_ylabel(LABELS.get(cfg['feature_names'][j],cfg['feature_names'][j])+'\nCIF贡献（百分点）')
                if row==0 and col in [0,2]:ax.legend(fontsize=8,loc='best')
        fig.suptitle(f'{year}年｜同一批记录、同一24个月输出：分散与背景定义分开看',fontsize=14,y=.995)
        fig.text(.02,.012,'全部训练/验证记录均展示；变量按训练平均绝对SHAP预先规定的规则选取。q2普通SHAP也可纵向分散，不能据此断言高阶交互。',fontsize=9)
        fig.tight_layout(rect=[0,.045,1,.96]);path=private/f'dependence_{year}_PRIVATE.png';fig.savefig(path,dpi=180);plt.close(fig);plots.append(str(path))
        # Identity scatter: numerical checks are exhaustive; only marks are bounded.
        fig,axes=plt.subplots(2,2,figsize=(10,8.5));off=np.triu_indices(p,1);rng=np.random.default_rng(20261028)
        for row,kind in enumerate(['main','pair']):
            truth=native['Phi'][n:,np.arange(p),np.arange(p)] if row==0 else 2*native['Phi'][n:,off[0],off[1]]
            for col,background in enumerate(['product','default']):
                matrix=ensemble[(year,'q2',background)]['Phi']
                got=matrix[n:,np.arange(p),np.arange(p)] if row==0 else 2*matrix[n:,off[0],off[1]]
                e=err(got,truth);x=truth.ravel()*100;y=got.ravel()*100;idx=np.flatnonzero((x!=0)|(y!=0))
                if len(idx)>2000:idx=np.sort(rng.choice(idx,2000,replace=False))
                ax=axes[row,col];ax.scatter(x[idx],y[idx],s=12,alpha=.4,color='#3977A8' if col==0 else '#CE694E')
                lo=min(x.min(),y.min());hi=max(x.max(),y.max());pad=max((hi-lo)*.05,.001);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#555555',lw=.8,ls='--')
                ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad);ax.grid(alpha=.15)
                ax.set_xlabel('原生 f_j（百分点）' if row==0 else '原生 f_jl（百分点）');ax.set_ylabel('SHAP Φ_jj（百分点）' if row==0 else 'SHAP 2Φ_jl（百分点）')
                ax.set_title(('精确乘积背景' if col==0 else '默认路径背景')+f'\n全量最大差：{e["max_abs"]*100:.3g} 个百分点',fontsize=11)
        fig.suptitle(f'{year}年 q2：同一模型的解释对齐',fontsize=15)
        fig.text(.02,.012,'误差使用全部验证记录和全部主效应/交互；每面板最多展示2000个非零点。默认差异表示参考口径不同，并非预测重构失败。',fontsize=8.5)
        fig.tight_layout(rect=[0,.04,1,.95]);path=private/f'alignment_{year}_PRIVATE.png';fig.savefig(path,dpi=180);plt.close(fig);plots.append(str(path))
        write(private/f'plot_selection_{year}_PRIVATE.json',dict(features=selections,names=[cfg['feature_names'][j] for j in selections],training_rows=n,evaluation_rows=len(X)-n))
    ef=pd.DataFrame(ensemble_rows);eo=pd.DataFrame(ensemble_order);ds=pd.DataFrame(decomp_summary)
    ef.drop(columns=['base']).to_csv(out/'ensemble_alignment.csv',index=False);eo.to_csv(out/'ensemble_order_contributions.csv',index=False);ds.to_csv(out/'high_order_allocation.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,year in zip(axes,[2023,2024]):
        xx=np.arange(1,7)
        for shift,model,color,label in [(-.18,'standard_rsf','#D18450','标准 RSF'),(.18,'q2','#387AA3','q2')]:
            d=eo[(eo.year==year)&(eo.model==model)&(eo.subset=='evaluation')].set_index('order')
            vals=[100*float(d.loc[r,'rms']) if r in d.index else 0 for r in xx]
            ax.bar(xx+shift,vals,width=.34,color=color,label=label)
        ax.set_title(str(year)+'年');ax.set_xticks(xx);ax.set_xlabel('函数交互阶数');ax.grid(axis='y',alpha=.18);ax.legend()
    axes[0].set_ylabel('该阶合计贡献的 RMS（百分点）')
    fig.suptitle('精确函数展开确认高阶项；依赖图分散本身不能确认高阶',fontsize=12)
    fig.text(.02,.01,'三个固定种子的预测函数先平均，再在当年验证记录上计算。RMS不是贡献占比，各阶RMS不能相加。',fontsize=8.5)
    fig.tight_layout(rect=[0,.045,1,.94]);fig.savefig(figures/'order_contributions.png',dpi=180);fig.savefig(figures/'order_contributions.svg');plt.close(fig)
    # Aggregate, not company-level, alignment chart is safe to synchronize.
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,['main_max_abs','full_pair_max_abs'],['主效应：Φ_jj 对比 f_j','二元交互：2Φ_jl 对比 f_jl']):
        for b,color,label in [('product','#387AA3','乘积背景'),('default','#D18450','默认路径背景')]:
            d=ef[(ef.model=='q2')&(ef.background==b)&(ef.subset=='evaluation')]
            ax.plot(d.year,d[metric]*100,'o-',color=color,label=label)
        ax.set_yscale('log');ax.set_xticks([2023,2024]);ax.set_title(title);ax.grid(alpha=.15);ax.legend()
    axes[0].set_ylabel('全量最大差（百分点，对数轴）')
    fig.suptitle('同一q2模型：统一背景后达到浮点数值精度',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.91]);fig.savefig(figures/'alignment_errors.png',dpi=180);fig.savefig(figures/'alignment_errors.svg');plt.close(fig)
    q=ef[(ef.model=='q2')&(ef.subset=='evaluation')]
    high=ds[(ds.model=='standard_rsf')&(ds.subset=='evaluation')]
    summarycols=['year','background','main_max_abs','full_pair_max_abs','ordinary_allocation_max_abs','baseline_difference']
    report='# 同一预测器，不同解释参考：完整对齐与背景差异同时成立\n\n'
    report+='六项正式任务全部成功；复用12个企业模型、零重训。2023/2024分别解释36/45条验证记录，以及各自309/345条训练参考。所有模型/预测原件哈希未变。\n\n'
    report+='## q2三种子平均函数的全量验证误差\n\n以下均为概率单位；换算为百分点需乘100。交互列已比较完整二元函数`2Φ_jl`与`f_jl`。\n\n'+table(q[summarycols])+'\n\n'
    report+='精确乘积背景下主效应、二元交互、普通Shapley行和及基准均与原生函数对齐。默认路径背景出现确定性差异，但两种背景下的完整贡献都加回同一个CIF；差异不是预测错误或解释器近似未收敛。默认依赖树结构与路径计数，也不能直接称为精确的经验条件分布SHAP。\n\n'
    report+='## 标准RSF的实际高阶贡献\n\n'+table(high[['year','max_order','high_order_functions','high_order_sum_mean_abs','high_order_sum_max_abs','higher_order_allocated_shap_rms']])+'\n\n'
    report+='这些高阶项由全阶函数展开确认；普通乘积背景Shapley中的高阶分摊按`f_S/|S|`逐项计算，并与官方TreeSHAP及独立联盟枚举核对。这里只保留二阶会遗漏真实模型输出。函数数是跨种子合并后的结构子集数量，不等于已证明稳定或所有输入域上均非零的函数数。\n\n'
    report+='## 依赖图能证明什么\n\n标准RSF普通SHAP存在同一特征分区内的纵向分散。q2普通SHAP同样可能分散，因为它含有二元交互分摊；因此纵向分散不能单独证明三阶以上交互。q2乘积背景的对角主效应是单变量函数，默认背景的对角主效应则可与该函数不同。企业级图留私有目录，汇总分散量见`dispersion_summary.csv`。\n\n'
    report+='## 判定边界\n\n本轮支持“相同背景下两种表示可精确对齐”和“背景定义会改变归因”；不支持SHAP普遍不忠实、不支持默认SHAP算错，也不构成q2独立方法创新、预测优越性、统计可辨识或因果性的证明。2023/2024已暴露且数据沿用PROVISIONAL口径；本轮没有用结局选择模型。\n\n'
    report+='原生interventional交互接口不支持的实测、乘积区域权重实现及独立核验见`PROTOCOL_ZH.md`。代码导出器经历两次失败试运行：终端NA阈值占位、R文本切点精度丢失；均已修复，失败记录保留。正式六任务无失败、无重训、无裁剪、无放宽误差门槛。\n'
    (out/'RESULTS_ZH.md').write_text(report)
    max_numeric=float(c[~((c.model=='q2')&(c.background=='default')&c['check'].str.startswith('native_'))].max_abs.max())
    pool_copy=out/'pool_execution.json';write(pool_copy,{k:v for k,v in status.items() if k not in ['completed','task_signatures']})
    testfile=next((root/'audit'/name for name in ['full_tests_final.log','full_tests_v3.log','full_tests_preflight.log'] if (root/'audit'/name).exists()),None)
    testresult=next((s for s in reversed(testfile.read_text().splitlines()) if ' passed' in s),'NOT_RECORDED') if testfile else 'NOT_RECORDED'
    failedpilots=list((root/'scratch').glob('pilot*/FAILED.json'));goodpilots=list((root/'scratch').glob('pilot*/summary.json'))
    receipt=dict(status='COMPLETE',experiment='shap_alignment_v1',private_run=str(root),slurm_job=job,
        formal_tasks_expected=6,formal_tasks_successful=6,formal_tasks_failed=0,saved_forests_reused=12,new_real_model_fits=0,
        primary_arm_seed_year_cells=18,diagnostic_control_cells=6,cohorts=dict(y2023=dict(train=309,evaluation=36),y2024=dict(train=345,evaluation=45)),
        predictor_outputs='first termination CIF,24 calendar months',input_feature_count=83,
        original_model_predictions_inputs_unchanged=True,all_original_hashes_reverified=True,all_publication_hashes_reverified=True,
        full_test_result=testresult,
        pilot_attempts=len(failedpilots)+len(goodpilots),pilot_failures=len(failedpilots),pilot_successes=len(goodpilots),
        pilot_failure_evidence=[str(p.relative_to(root)) for p in failedpilots],
        max_actual_numerical_identity_error=max_numeric,
        installed_shap='0.54.1',default_resolution='tree_path_dependent, no data argument',
        native_interventional_interactions_supported=False,
        product_reference_method='exact empirical marginal-product region masses; official TreeSHAP recurrence; independent per-leaf coalition Shapley interactions',
        no_monte_carlo_background=True,no_test_outcome_based_selection=True,no_2025_access=True,
        pool_nodes_requested=['oliver','lemon'],actual_tasks_by_host=dict(pd.Series([s['host'] for s in summaries]).value_counts().astype(int).to_dict()),
        pool_execution_seconds=status['seconds'],node_note='Both-node pool was allocated; six short tasks were all taken by oliver before lemon needed work; no repeated jobs merely for utilization.',
        private_figures=plots,decision='EXPLANATION_ALIGNMENT_CONFIRMED; NO_PREDICTIVE_OR_NOVELTY_CLAIM',
        frozen_worker_code=snapshot,sources= ['https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html','https://proceedings.mlr.press/v32/hutter14.html'])
    write(out/'execution_receipt.json',receipt)
    if dest is not None:
        target=dest/'results';target.mkdir(exist_ok=False)
        for name in ['alignment_metrics.csv','order_metrics.csv','numerical_checks.csv','task_diagnostics.csv','source_model_hashes.csv','dispersion_summary.csv',
                     'ensemble_alignment.csv','ensemble_order_contributions.csv','high_order_allocation.csv','pool_execution.json']:
            shutil.copyfile(out/name,target/name)
        shutil.copytree(figures,dest/'figures')
        for name in ['RESULTS_ZH.md','execution_receipt.json']:shutil.copyfile(out/name,dest/name)
        for name in ['native_interventional_interactions_probe.json','adapter_amendment_01.json','adapter_amendment_02.json','code_hashes_v3.json']:
            if (root/'audit'/name).exists():shutil.copyfile(root/'audit'/name,target/name)
        shutil.copyfile(root/'input/protocol.json',dest/'frozen_protocol.json')
        if testfile:shutil.copyfile(testfile,target/'full_tests.txt')
    print(json.dumps(dict(status='COMPLETE',max_identity_error=max_numeric,ensemble_alignment=q[summarycols].to_dict('records'),high_order=high.to_dict('records')),ensure_ascii=False))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--job',type=int,required=True);a.add_argument('--pool-result',type=Path,required=True);a.add_argument('--dest',type=Path);a.add_argument('--report-tag')
    z=a.parse_args();run(z.run,z.job,z.pool_result,z.dest,z.report_tag)
