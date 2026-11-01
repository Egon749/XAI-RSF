"""Aggregate fixed-model evidence and cautious, prespecified feature triage."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
from .common import *


def ci(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return list(map(float,np.quantile(x,[.025,.975]))) if len(x) else [np.nan,np.nan]


def retention(row, component=False):
    prefix='component_' if component else ''
    return bool(row['cohorts']>=2 and row[prefix+'positive_cohorts']>=2 and row[prefix+'delta_auc']>=.01 and row[prefix+'auc_lo']>0)


def analyze(run):
    gate=json.loads((run/'audit/formal_gate.json').read_text());assert gate['status']=='PASS'
    rows=[];baseline=[];components=[];pairs=[];summaries=[];bootstrap={};iboot={}
    for i in range(gate['tasks']):
        p=run/'results/formal'/str(gate['job'])/f'task-{i}/output'
        d=pd.read_csv(p/'effects.csv',dtype={'key':str,'moved':str});z=arrays(p/'bootstrap_PRIVATE.npz')
        for r in d.to_dict('records'):
            uid=len(rows);index=r.pop('bootstrap_row');r['uid']=uid;rows.append(r);bootstrap[uid]=z['effects'][index];iboot[uid]=z['component_auc'][index]
        baseline.append(pd.read_csv(p/'baseline.csv'))
        for file,dest in [('components.csv',components),('pair_support.csv',pairs)]:
            try:dest.append(pd.read_csv(p/file,dtype={'key':str,'moved':str}))
            except pd.errors.EmptyDataError:pass
        summaries.append(json.loads((p/'summary.json').read_text()))
    d=pd.DataFrame(rows);out=run/'reports';d.to_csv(out/'permutation_by_cohort.csv',index=False)
    base=pd.concat(baseline).drop_duplicates();base.to_csv(out/'baseline_metrics.csv',index=False)
    comp=pd.concat(components);comp.to_csv(out/'native_function_contributions.csv',index=False)
    stability=[]
    for (h,cause,key),g in comp[comp.kind=='pair'].groupby(['horizon','cause','key']):
        row=dict(horizon=h,cause=cause,key=key,label=g.label.iloc[0],structural_cohorts=len(g),
            nonzero_cohorts=int((g.train_rms>1e-14).sum()),validation_nonzero_cohorts=int((g.validation_rms>1e-14).sum()))
        for year in range(2019,2024):
            a=g[g.cohort==year];row[f'train_rms_{year}']=float(a.train_rms.iloc[0]) if len(a) else 0.
        stability.append(row)
    pd.DataFrame(stability).to_csv(out/'interaction_stability.csv',index=False)
    pair=pd.concat(pairs);pair.to_csv(out/'interaction_support.csv',index=False)
    agg=[]
    for keys,g in d[d.informative].groupby(['outer_cutoff','horizon','cause','kind','key']):
        outer,h,c,kind,key=keys;assert not g.cohort.duplicated().any()
        b=np.stack([bootstrap[i] for i in g.uid]);a=np.stack([iboot[i] for i in g.uid])
        means=np.mean(b,axis=0);am=np.mean(a,axis=0)
        row=dict(outer_cutoff=outer,horizon=h,cause=c,kind=kind,key=key,label=g.label.iloc[0],moved=g.moved.iloc[0],linked_group=bool(g.linked_group.any()),
            cohorts=len(g),cohort_years=','.join(map(str,sorted(g.cohort))),companies=int(g.n.sum()),events=int(g.validation_events.sum()),
            minimum_train_events=int(g.train_events.min()),positive_cohorts=int((g.delta_auc>=.005).sum()),
            delta_auc=float(g.delta_auc.mean()),delta_ap=float(g.delta_ap.mean()),delta_brier=float(g.delta_brier.mean()),
            auc_bootstrap_undefined=int((~np.isfinite(means[:,0])).sum()))
        for j,k in enumerate(['auc','brier']):row[k+'_lo'],row[k+'_hi']=ci(means[:,j])
        if kind=='pair':
            row.update(component_delta_auc=float(g.component_delta_auc.mean()),component_delta_ap=float(g.component_delta_ap.mean()),component_positive_cohorts=int((g.component_delta_auc>=.005).sum()))
            row['component_auc_lo'],row['component_auc_hi']=ci(am)
            row['component_retention_evidence']=retention(row,True)
        else:row['component_retention_evidence']=False
        row['retention_evidence']=retention(row);agg.append(row)
    ag=pd.DataFrame(agg);ag.to_csv(out/'permutation_temporal_summary.csv',index=False)
    # Joint pair importance includes its mains. This contrast is descriptive
    # nonadditivity, NOT an isolated causal interaction or a SHAP interaction.
    contrasts=[]
    for r in d[d.kind=='pair'].itertuples():
        j,k=map(int,r.key.split(','));g=d[(d.outer_cutoff==r.outer_cutoff)&(d.cohort==r.cohort)&(d.horizon==r.horizon)&(d.cause==r.cause)&(d.kind=='single')]
        s=g.set_index('key');contrasts.append(dict(outer_cutoff=r.outer_cutoff,cohort=r.cohort,horizon=r.horizon,cause=r.cause,key=r.key,
            joint_delta_auc=r.delta_auc,nonadditivity_delta_auc=r.delta_auc-s.loc[str(j),'delta_auc']-s.loc[str(k),'delta_auc'],linked_group_overlap=bool(set(logical_closure([j]))&set(logical_closure([k])))))
    pd.DataFrame(contrasts).to_csv(out/'pair_joint_vs_single.csv',index=False)
    features=json.loads((run/'input/features.json').read_text());quality=pd.read_csv(out/'feature_quality.csv');edges=pd.read_csv(out/'correlation_edges.csv')
    classification=[];stable_edges=[]
    for outer in [2023,2024]:
        q=quality[quality.cohort<outer];edge=edges[edges.cohort<outer]
        candidates={}
        for (j,k),g in edge.groupby(['j','k']):
            if g.cohort.nunique()<3:continue
            mj=q[q.feature==j].missing_rate.mean();mk=q[q.feature==k].missing_rate.mean()
            keep=j if (mj,j)<=(mk,k) else k;other=k if keep==j else j
            candidates.setdefault(other,[]).append(keep)
            stable_edges.append(dict(outer_cutoff=outer,j=j,k=k,label_j=features[j]['label'],label_k=features[k]['label'],cohorts=len(g),min_abs_rho=float(abs(g.rho).min()),representative=keep,redundancy_candidate=other))
        for f in features:
            j=f['index'];category='证据不足';subtype='low_support_or_unresolved';reason='未达到跨年份保留或疑似噪声证据门槛；不等于无预测价值';protection=[]
            for cause in [1,2]:
                a=ag[(ag.outer_cutoff==outer)&(ag.horizon==24)&(ag.cause==cause)]
                direct=a[(a.kind=='single')&(a.key==str(j))&a.retention_evidence]
                group=a[(a.kind=='group')&a.retention_evidence]
                group=group[group.key.map(lambda k:j in map(int,k.split(',')))]
                interactions=a[(a.kind=='pair')&a.component_retention_evidence]
                interactions=interactions[interactions.key.map(lambda k:j in map(int,k.split(',')))]
                if len(direct) or len(group) or len(interactions):
                    protection.append(('退市' if cause==1 else '恢复')+('成组置换' if len(group) or (len(direct) and direct.linked_group.any()) else '单变量置换' if len(direct) else '原生交互分数诊断'))
            own=q[q.feature==j]
            if protection:
                category='保留';subtype='provisional_predictive_or_group_evidence';reason='；'.join(protection)+'跨至少两个历史窗口达到预设门槛；仍属探索性'
            elif own.constant.all():
                category='疑似冗余或噪声';subtype='constant_in_available_training';reason='各可用历史训练窗口均为常量（含缺失状态）；仅说明这些训练窗口无区分能力'
            elif j in candidates:
                category='疑似冗余或噪声';subtype='suspected_redundant';reason='与'+ '、'.join(features[k]['label'] for k in sorted(set(candidates[j])))+'在至少三个训练窗口|Spearman|≥0.95；只是可替代候选，尚未删列重训'
            else:
                a=ag[(ag.outer_cutoff==outer)&(ag.horizon==24)&(ag.cause==1)&(ag.kind=='single')&(ag.key==str(j))]
                if len(a) and a.iloc[0].cohorts>=2 and a.iloc[0].minimum_train_events>=10 and a.iloc[0].auc_hi<.01 and a.iloc[0].brier_hi<.001:
                    category='疑似冗余或噪声';subtype='suspected_noise';reason='充分支持的历史窗口中达到预设近零增量界限；仍需删列重训核验'
            classification.append(dict(outer_cutoff=outer,feature=j,name=f['name'],label=f['label'],category=category,subtype=subtype,reason=reason,
                train_missing_mean=float(own.missing_rate.mean()),used_in_split_trees=int(own.split_trees.sum()),training_windows=len(own)))
    pd.DataFrame(classification).to_csv(out/'feature_classification.csv',index=False)
    pd.DataFrame(stable_edges).to_csv(out/'stable_redundancy_candidates.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'execution_tasks.csv',index=False)
    assert len(classification)==166 and all(s['new_fits']==0 for s in summaries)
    for p,h in json.loads((run/'audit/source_hashes.json').read_text()).items():assert sha(Path(p))==h
    for p,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/p)==h
    repo=Path(__file__).resolve().parents[2];assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    write(run/'audit/analysis_validation.json',dict(status='PASS',tasks=len(summaries),new_fits=0,source_and_input_hashes_unchanged=True,
        current_reference_unchanged=True,maximum_permutation_error=max(s['maximum_exact_permutation_error'] for s in summaries),
        unique_items=sum(s['items'] for s in summaries),metric_rows=len(d),classification_rows=len(classification)))
    print(pd.DataFrame(classification).groupby(['outer_cutoff','category']).size().to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run)
