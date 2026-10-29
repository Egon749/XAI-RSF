from pathlib import Path
import argparse,json,os,resource,socket,time,traceback
import numpy as np
from crforest import CompetingRiskForest,ForestConfig
from crforest.shrinkage import shrink_leaf_cifs
from crforest.temporal_leaf import temporal_leaf_cifs
from experiments.post_st_prognosis_v1.common import arrays,sha,write
from experiments.post_st_leaf_shallow_v1.worker import emit


def run(a):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));started=time.perf_counter();cpu=time.process_time()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text());identity=json.loads((root/'code_identity.json').read_text())
    src=root/'validation'/str(a.validation_year) if a.stage=='validation' else root/'formal'/str(a.year)
    tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz');assert set(ev)=={'X','sample_ids'}
    cfg=json.loads((src/'config.json').read_text());cfg.update(seed=a.seed,n_trees=4 if a.stage=='pilot' else 32)
    fitted=0;fit_seconds=0.
    if a.stage=='formal':
        f=CompetingRiskForest.load(src/'reference'/str(a.seed)/'shallow/forest_PRIVATE.json')
        assert f.config==ForestConfig(**cfg)
        baseline=CompetingRiskForest.load(src/'reference'/str(a.seed)/'shallow_parent/forest_PRIVATE.json')
        control=temporal_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],np.ones(len(tr['X'])),10)
        np.testing.assert_array_equal(control.predict(np.vstack([tr['X'],ev['X']])),baseline.predict(np.vstack([tr['X'],ev['X']])))
        original=arrays(src/'reference'/str(a.seed)/'shallow_parent/predictions_PRIVATE.npz')
        np.testing.assert_array_equal(baseline.predict(ev['X'],original['times']),original['cif'])
    else:
        start=time.perf_counter();f=CompetingRiskForest(ForestConfig(**cfg)).fit(tr['X'],tr['time'],tr['event'],tr['groups'])
        fit_seconds+=time.perf_counter()-start;fitted+=1
        baseline=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10)
    models={'baseline':baseline};update_seconds=0.
    for w in [2,4]:
        weights=np.where(tr['year']>=protocol['recent_entry_year_start'],float(w),1.)
        start=time.perf_counter();models[f'time_w{w}']=temporal_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],weights,10)
        update_seconds+=time.perf_counter()-start
        assert models[f'time_w{w}'].audit_==f.audit_
    start=time.perf_counter();larger=CompetingRiskForest(ForestConfig(**{**cfg,'min_leaf':20})).fit(tr['X'],tr['time'],tr['event'],tr['groups'])
    fit_seconds+=time.perf_counter()-start;fitted+=1
    models['leaf20']=shrink_leaf_cifs(larger,tr['X'],tr['time'],tr['event'],tr['groups'],10)
    for arm,model in models.items():
        meta=dict(stage=a.stage,test_year=a.year,validation_year=a.validation_year,seed=a.seed,pilot=a.stage=='pilot',
            host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0)),code_commit=identity['commit'],
            source_train_sha256=sha(src/'train_PRIVATE.npz'),base_fit_seconds=fit_seconds,
            recent_case_weight={'baseline':1,'time_w2':2,'time_w4':4,'leaf20':1}[arm],min_leaf=model.config.min_leaf)
        emit(model,tr,ev,out/arm,arm,meta)
    summary=dict(status='COMPLETE',stage=a.stage,year=a.year,validation_year=a.validation_year,seed=a.seed,
        new_forest_fits=fitted,model_artifacts=len(models),host=socket.gethostname(),code_commit=identity['commit'],
        seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,fit_seconds=fit_seconds,
        weighted_update_seconds=update_seconds,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        no_evaluation_labels_loaded=True,baseline_reproduced=a.stage=='formal')
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','validation','formal'],required=True)
    p.add_argument('--year',type=int,required=True);p.add_argument('--validation-year',type=int,default=0);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
