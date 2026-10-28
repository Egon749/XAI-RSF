"""Compare fixed old/new predictions on the identical 2024 company cohort."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from crforest.core import aalen_johansen
from crforest.partition import tree_feature_subset
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizon_predictions
from experiments.post_st_prognosis_v1.validate import direct_metrics
from .prepare import BASE


def compare(run, output):
    output.mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    old = repo/'experiments/post_st_prognosis_v1/results'
    new = run/'evaluation'
    receipt = json.loads((old.parent/'execution_receipt.json').read_text())
    validation = json.loads((new/'validation.json').read_text())
    assert validation['status'] == 'PASS'
    for name in ['metrics.csv', 'metrics_summary.csv', 'warning_metrics.csv', 'interpretability.csv', 'seed_stability.csv']:
        assert sha(old/name) == receipt['public_results_sha256'][name]
    plan = json.loads((run/'input/plan.json').read_text())
    prior_plan = json.loads((BASE/'input/plan.json').read_text())
    for key in ['forest_common', 'forest_variants', 'standard_rsf', 'seeds', 'variants', 'feature_names']:
        assert plan[key] == prior_plan[key], key
    oldev = arrays(BASE/'input/eval_PRIVATE.npz')
    newev = arrays(run/'input/eval_PRIVATE.npz')
    mask = oldev['year'] == 2024
    for key in newev:
        np.testing.assert_array_equal(newev[key], oldev[key][mask])
    old_predictions = {}
    support = []
    price = plan['feature_names'].index('log_price_last')
    config_by_key = {}
    for version, root, records in [('old', BASE, receipt['model_hashes']), ('new', run, validation['model_hashes'])]:
        hashes = {(r['model'], r['seed']):r for r in records}
        for path in sorted((root/'results/formal').glob('*/task-*/output/summary.json')):
            summary = json.loads(path.read_text()); directory = path.parent
            key = (summary['model'], summary['seed']); record = hashes[key]
            assert sha(directory/record['model_file']) == record['model_sha256']
            assert sha(directory/'predictions_PRIVATE.npz') == record['prediction_sha256']
            if version == 'old':
                p = arrays(directory/'predictions_PRIVATE.npz')
                np.testing.assert_array_equal(p['sample_ids'], oldev['sample_ids'])
                old_predictions[(key[0], str(key[1]))] = dict(cif=p['cif'][mask], times=p['times'])
            if key[0] == 'standard_rsf':
                continue
            f = json.loads((directory/'forest_PRIVATE.json').read_text()); cfg = f['config']
            if version == 'old':config_by_key[key] = cfg
            else:assert cfg == config_by_key[key]
            leaves = [n for tree in f['trees'] for n in tree if 'cif' in n]
            used = {n['feature'] for tree in f['trees'] for n in tree if 'feature' in n}
            if version == 'new':
                write(run/'private'/f'used_inputs_{key[0]}_{key[1]}_PRIVATE.json',
                      [plan['feature_names'][j] for j in sorted(used)])
            support.append(dict(version=version, model=key[0], seed=key[1], trees=len(f['trees']),
                actual_used_inputs=len(used), leaves=len(leaves),
                zero_termination_leaves=sum(n['cause_company_support'][0] == 0 for n in leaves),
                fewer_than_three_termination_leaves=sum(n['cause_company_support'][0] < 3 for n in leaves),
                zero_termination_leaf_fraction=np.mean([n['cause_company_support'][0] == 0 for n in leaves]),
                fewer_than_three_termination_leaf_fraction=np.mean([n['cause_company_support'][0] < 3 for n in leaves]),
                price_eligible_trees=sum(price in tree_feature_subset(f['n_features'], cfg['tree_feature_count'], cfg['seed'], i) for i in range(len(f['trees']))),
                price_used_trees=sum(any(n.get('feature') == price for n in tree) for tree in f['trees']),
                root_price_trees=sum(tree[0].get('feature') == price for tree in f['trees']),
                root_termination_company_min=min(t[0]['cause_company_support'][0] for t in f['trees']),
                root_termination_company_max=max(t[0]['cause_company_support'][0] for t in f['trees'])))
    # Independently verify prior reported 2024 scores against saved predictions,
    # with the very same endpoint rows as the new run.
    for model in plan['variants']:
        pp = [old_predictions[(model, str(seed))] for seed in plan['seeds']]
        old_predictions[(model, 'ensemble')] = dict(times=pp[0]['times'], cif=np.mean([p['cif'] for p in pp], axis=0))
    tr = arrays(BASE/'input/train_PRIVATE.npz'); times = np.array([365.,366.,730.,731.])
    baseline = aalen_johansen(tr['time'], tr['event'], times)
    old_predictions[('training_AJ', 'none')] = dict(times=times, cif=np.broadcast_to(baseline, (45, 4, 2)))
    prior_metrics = pd.read_csv(old/'metrics.csv', dtype={'scope':str, 'seed':str})
    checked = 0
    for row in prior_metrics[prior_metrics.scope == '2024'].itertuples():
        p = old_predictions[(row.model, row.seed)]; h = newev[f'h{row.horizon_months}']
        k = 1 if row.cause == 'termination' else 2
        y = (newev['event'] == k) & (newev['time'] <= h)
        risk = horizon_predictions(p['cif'], p['times'], h)[:,k-1]
        np.testing.assert_allclose(direct_metrics(y,risk), [row.auc,row.ap,row.brier], rtol=0, atol=5e-14)
        checked += 1
    assert checked == 52
    keys = ['model','scope','horizon_months','cause']
    panels = []
    for version, path in [('old',old), ('new',new)]:
        frame = pd.read_csv(path/'metrics_summary.csv', dtype={'scope':str})
        panels.append(frame[frame.scope == '2024'].assign(version=version))
    long = pd.concat(panels, ignore_index=True)
    long.to_csv(output/'seed_mean_comparison.csv', index=False)
    left, right = [p.drop(columns='version').set_index(keys) for p in panels]
    delta = right[['auc_mean','ap_mean','brier_mean','risk_mean_mean']] - left[['auc_mean','ap_mean','brier_mean','risk_mean_mean']]
    delta.add_prefix('new_minus_old_').reset_index().to_csv(output/'metric_deltas.csv', index=False)
    for name in ['metrics.csv', 'warning_metrics.csv', 'interpretability.csv', 'seed_stability.csv']:
        frames=[]
        for version,path in [('old',old),('new',new)]:
            f=pd.read_csv(path/name, dtype={'scope':str,'seed':str})
            if 'scope' in f:f=f[f.scope == '2024']
            frames.append(f.assign(version=version))
        pd.concat(frames,ignore_index=True).to_csv(output/f'old_new_{name}',index=False)
    pd.DataFrame(support).to_csv(output/'tree_support_and_price_use.csv', index=False)
    write(output/'comparison_receipt.json',dict(status='PASS',identical_test_companies=45,
        old_prediction_metric_rows_independently_checked=52,new_prediction_metric_rows_independently_checked=52,
        matching_forests_and_hyperparameters=True,new_fits_in_comparison=0,
        old_metrics_source_sha256=sha(old/'metrics.csv'),new_metrics_source_sha256=sha(new/'metrics.csv'),
        note='Same seed labels, but bootstrap samples differ with training population; seed ranges are not confidence intervals.'))
    plot(long,output)
    print(delta.reset_index().to_string(index=False))


def plot(frame, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,'svg.hashsalt':'train2023-v1'})
    models=['xai_q2','unrestricted','standard_rsf'];labels=['XAI-RSF q2','同后端无限阶','标准RSF']
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    colors={'old':'#819bb6','new':'#d67548'}
    for row,horizon in enumerate([24,12]):
        part=frame[(frame.horizon_months==horizon)&(frame.cause=='termination')]
        for col,metric in enumerate(['auc','ap','brier']):
            ax=axes[row,col]
            for version,shift in [('old',-.17),('new',.17)]:
                d=part[part.version==version].set_index('model').loc[models]
                mean=d[f'{metric}_mean'].to_numpy();err=np.stack([mean-d[f'{metric}_min'],d[f'{metric}_max']-mean])
                bars=ax.bar(np.arange(3)+shift,mean,.32,yerr=err,capsize=3,color=colors[version],
                    label='原训练' if version=='old' else '扩充至2023')
                ax.bar_label(bars,labels=[f'{v:.3f}' for v in mean],padding=4,fontsize=8)
            ax.set_xticks(np.arange(3),labels,fontsize=9)
            ax.set_ylim(0,1 if metric!='brier' else .48)
            ax.set_title(f'{horizon}个月退市 · '+{'auc':'AUC ↑','ap':'AP ↑','brier':'Brier ↓'}[metric])
            ax.spines[['top','right']].set_visible(False)
            if row==0 and col==0:ax.legend(fontsize=9)
    fig.suptitle('2024年同一45家公司：训练信息扩充前后',fontsize=15)
    fig.text(.5,.018,'柱为三个固定种子的指标均值，误差线为最小—最大值，并非置信区间。\n'
             '未更改模型配置；主终点24个月，12个月为补充。已暴露结果的探索性比较。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(out/'prediction_change.png',dpi=170)
    fig.savefig(out/'prediction_change.svg',metadata={'Date':None});plt.close(fig)
    p=out/'prediction_change.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();compare(a.run,a.output)
