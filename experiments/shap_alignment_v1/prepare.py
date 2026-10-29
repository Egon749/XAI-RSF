"""Freeze feature-only inputs, saved models, hypotheses and finite compute tasks."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from experiments.post_st_leaf_shallow_v1.analyze import artifacts


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def prepare(run,repo):
    current=json.loads((repo/'CURRENT_POST_ST_MODEL.json').read_text())
    old=Path(current['private_run']);sources={};inputs=run/'input';configs=[]
    paths={(y,a,str(s)):p for y,a,s,p in artifacts(old)}
    modelcheck={(m['test_year'],str(m['seed'])):m for m in current['saved_models']}
    def copy(src,dst):
        src=Path(src);sources[str(src)]=sha(src);shutil.copyfile(src,dst);assert sha(dst)==sources[str(src)]
    for y in [2023,2024]:
        trpath=old/'input/folds'/str(y)/'train_PRIVATE.npz';tepath=old/'evaluation_input'/f'{y}_test_PRIVATE.npz'
        for p in [trpath,tepath]:sources[str(p)]=sha(p)
        with np.load(trpath,allow_pickle=False) as z:train={k:z[k] for k in ['X','sample_ids']}
        with np.load(tepath,allow_pickle=False) as z:test={k:z[k] for k in ['X','sample_ids','h24']}
        horizons=np.unique(test['h24']);assert len(horizons)==1
        names=json.loads((old/'input/folds'/str(y)/'plan.json').read_text())['feature_names']
        for seed in [42,101,202]:
            dst=inputs/str(y)/str(seed);dst.mkdir(parents=True)
            np.savez_compressed(dst/'features_PRIVATE.npz',train_X=train['X'],test_X=test['X'],
                                train_ids=train['sample_ids'],test_ids=test['sample_ids'])
            q=paths[(y,'shallow_parent',str(seed))];r=paths[(y,'standard_rsf',str(seed))]
            assert sha(q/'forest_PRIVATE.json')==modelcheck[(y,str(seed))]['forest_sha256']
            assert sha(q/'functions_PRIVATE.npz')==modelcheck[(y,str(seed))]['functions_sha256']
            for src,name in [(q/'forest_PRIVATE.json','q2_forest_PRIVATE.json'),(q/'functions_PRIVATE.npz','q2_functions_PRIVATE.npz'),
                             (q/'predictions_PRIVATE.npz','q2_predictions_PRIVATE.npz'),(r/'forest_PRIVATE.rds','rsf_forest_PRIVATE.rds'),
                             (r/'predictions_PRIVATE.npz','rsf_predictions_PRIVATE.npz'),(r/'training_imputation_PRIVATE.npy','rsf_imputation_PRIVATE.npy')]:
                copy(src,dst/name)
            cfg=dict(year=y,seed=seed,horizon_days=int(horizons[0]),p=len(names),n_train=len(train['X']),n_test=len(test['X']),
                     feature_names=names,reference='exact product of empirical frozen training marginals',
                     rsf_reference='training-only median-imputed input; q2 retains missing states')
            write(dst/'config.json',cfg);configs.append(cfg)
    protocol=dict(experiment='shap_alignment_v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        model_state=current['status'],source_commit='5262a37793fd41c821badd3bb9e1a2aec4e873b2',
        cohorts=[2023,2024],seeds=[42,101,202],new_model_fits=0,calibration_fits=0,
        output='24-calendar-month first termination CIF: day731 in2023/day730 in2024; scalar raw adapter equals CIF',
        explanation_rows='all frozen training and held-out feature rows; no outcome labels used for explanations or selection',
        primary_arms=['standard_rsf_default_treeshap','q2_exact_product_treeshap_interactions','q2_default_treeshap'],
        diagnostic_control='standard_rsf_exact_product_treeshap: separates reference effects from high-order allocation',
        default_definition='shap.TreeExplainer(lossless_scalar_CIF_tree_dictionary), no background and no perturbation override; log installed version and resolved setting',
        product_reference='exact node-region masses under product empirical marginals; same leaf outputs and tree structure; no finite shuffled-row approximation',
        independent_check='enumerate leaf coalitions and factorial Shapley interaction weights; also compare q2 against saved native functions',
        numerical_tolerance_absolute=1e-12,
        high_order_check='all actual-order product-reference terms and their exact share f_S/|S| in ordinary Shapley; no inference from scatter alone',
        dispersion='within exact feature partition cells; q2 ordinary SHAP included as a negative control for the claim that scatter implies order>=3',
        plot_selection='per year choose leading standard-default and q2-product features by TRAINING mean absolute ordinary SHAP; never by outcomes or largest mismatch',
        figures='company-level dependence/identity plots remain private; aggregate numerical/error/order summaries may be committed',
        predictions_unchanged=True,no_2025=True,no_retraining=True,
        evidence='explanation semantics audit, not prediction improvement or independent test; inherited data status PROVISIONAL',
        resources=dict(independent_tasks=6,cpu_threads_per_task=1,memory_gib_per_task=4,timeout_seconds=1800,pool='existing two-node Slurm'))
    write(inputs/'protocol.json',protocol);write(run/'audit/source_hashes_PRIVATE.json',sources)
    write(run/'audit/configurations.json',configs)
    code=run/'code';shutil.copytree(repo/'crforest',code/'crforest',ignore=shutil.ignore_patterns('__pycache__'))
    (code/'experiments').mkdir();shutil.copytree(repo/'experiments/shap_alignment_v1',code/'experiments/shap_alignment_v1',ignore=shutil.ignore_patterns('__pycache__'))
    for file in [repo/'CURRENT_POST_ST_MODEL.json',repo/'pyproject.toml']:copy(file,code/file.name)
    write(run/'audit/input_hashes_PRIVATE.json',{str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()})
    write(run/'audit/code_hashes.json',{str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()})
    tasks=[]
    for cfg in configs:
        tasks.append(dict(kind='cpu',argv=['python3',str(code/'experiments/shap_alignment_v1/launch.py'),
            '--year',str(cfg['year']),'--seed',str(cfg['seed'])],cwd=str(code),timeout=1800,
            storage=dict(source=str(inputs),mode='nfs',results=str(run/'results'))))
    (run/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'worker.py').write_text('import runpy\nrunpy.run_path('+repr(str(code/'experiments/shap_alignment_v1/launch.py'))+', run_name="__main__")\n')
    write(run/'worker.py.hetero.json',dict(independent=True,manifest='tasks.jsonl'))
    write(run/'audit/prepared.json',dict(status='FROZEN',tasks=6,new_fits=0,protocol_sha256=sha(inputs/'protocol.json')))
    print(json.dumps(dict(run=str(run),tasks=len(tasks),status='FROZEN')))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--repo',type=Path,required=True)
    args=a.parse_args();prepare(args.run,args.repo)
