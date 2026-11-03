"""Disjoint tree growth, exact assembly, and complete prefix exports."""
import argparse,fcntl,os,resource,socket,time,traceback,hashlib
from contextlib import contextmanager
from crforest import CompetingRiskForest
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.worker import verify_export
from .nested import grow_range,assemble,prefix
from .common import *

@contextmanager
def memory_slot(stage,root):
    if stage=='grow':yield;return
    slots=2 if socket.gethostname()=='lemon'else 12
    key=hashlib.sha256(str(root).encode()).hexdigest()[:16]
    d=Path.home()/'.cache'/'xai-tree-count-slots'/key;d.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();held=None
    while held is None:
        for i in range(slots):
            f=(d/f'{i}.lock').open('a')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held=f;break
            except BlockingIOError:f.close()
        if held is None:
            if time.perf_counter()-start>3300:raise TimeoutError('memory slot wait')
            time.sleep(.1)
    try:yield
    finally:held.close()

def verify_saved(f,tr,ev,path):
    f.save(path);loaded=CompetingRiskForest.load(path);err=0.;minimum=0.;maximum=0.;increment=0.
    x=np.vstack([tr['X'],ev['X']]);times=np.unique(np.r_[0.,f.grid_,TIMES])
    for i in range(0,len(x),32):
        p=f.predict(x[i:i+32],times);q=loaded.predict(x[i:i+32],times)
        assert np.isfinite(p).all();err=max(err,float(abs(p-q).max()))
        minimum=min(minimum,float(p.min()));maximum=max(maximum,float(p.sum(-1).max()));increment=min(increment,float(np.diff(p,axis=1).min()))
    assert err<=1e-10 and minimum>=-1e-12 and maximum<=1+1e-12 and increment>=-1e-12
    return dict(reload_error=err,probability_min=minimum,probability_sum_max=maximum,minimum_increment=increment)

def main(a):
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    start=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    with memory_slot(a.mode,root):
        waited=time.perf_counter()-start;tr=arrays(root/f'folds/{a.year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{a.year}/eval_PRIVATE.npz')
        assert set(ev)=={'X','sample_ids'}and tr['X'].shape[1]==74
        detail={};registry=None
        if a.registry:
            assert sha(Path(a.registry))==a.registry_sha256;registry=load(a.registry)
        if a.mode=='grow':
            t=time.perf_counter();z=grow_range(configuration(a.seed,256),tr,a.start,a.stop)
            # Same JSON nonfinite-cut convention as the existing forest serializer.
            with (out/'trees_PRIVATE.json').open('x')as f:json.dump(z,f)
            detail=dict(new_trees=len(z),fit_seconds=time.perf_counter()-t,tree_start=a.start,tree_stop=a.stop,
                candidate_evaluations=sum(t['candidate_evaluations']for t in z))
        elif a.mode=='assemble':
            import pandas as pd
            source=Path(load(root/'old_registry.json')[f'{a.year}|{a.seed}'])
            expected=load(source/'hashes.json')
            for n,h in expected.items():assert sha(source/n)==h
            f=CompetingRiskForest.load(source/'raw/forest_PRIVATE.json')
            nd=pd.read_csv(source/'node_search_PRIVATE.csv')
            counts=nd.groupby('tree').routed_attempts.sum().reindex(range(32),fill_value=0).astype(int).tolist()
            chunks=[]
            for rec in registry:
                if rec['year']==a.year and rec['seed']==a.seed:
                    p=Path(rec['path']);assert sha(p/'trees_PRIVATE.json')==load(p/'hashes.json')['trees_PRIVATE.json']
                    chunks+=load(p/'trees_PRIVATE.json')
            f,counts=assemble(f,chunks,256,counts)
            assert json.dumps(f.trees_[:32])==json.dumps(CompetingRiskForest.load(source/'raw/forest_PRIVATE.json').trees_)
            t=time.perf_counter();s=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10);shr_seconds=time.perf_counter()-t
            old=CompetingRiskForest.load(source/'shrunk/forest_PRIVATE.json')
            assert json.dumps(s.trees_[:32])==json.dumps(old.trees_)
            checks={v:verify_saved(model,tr,ev,out/f'{v}_PRIVATE.json')for v,model in [('raw',f),('shrunk',s)]}
            write(out/'tree_counts_PRIVATE.json',counts)
            pairs=pairing(tr['groups'],f.config,74);write(out/'pairing_PRIVATE.json',pairs)
            for z in chunks:assert z['inbag_sha256']==pairs[z['tree_id']]['multiplicity_sha256']
            detail=dict(new_trees=0,assembled_trees=256,reused_prefix_trees=32,new_suffix_trees=224,
                old32_raw_identical=True,old32_shrunk_identical=True,shrink_seconds=shr_seconds,checks=checks)
        else:
            selected=[x for x in registry if x['year']==a.year and(a.seed==0 or x['seed']==a.seed)]
            assert len(selected)==(3 if a.seed==0 else 1);detail=dict(new_trees=0,checks={})
            for version in ['raw','shrunk']:
                members=[]
                for rec in sorted(selected,key=lambda x:x['seed']):
                    p=Path(rec['path']);hs=load(p/'hashes.json')
                    for n in [f'{version}_PRIVATE.json','tree_counts_PRIVATE.json']:assert sha(p/n)==hs[n]
                    f=CompetingRiskForest.load(p/f'{version}_PRIVATE.json')
                    members.append(prefix(f,a.count,load(p/'tree_counts_PRIVATE.json')))
                f=merge(members)if a.seed==0 else members[0]
                assert len(f.trees_)==a.count*len(members)
                error=float(abs(f.predict(ev['X'],TIMES)-np.mean([m.predict(ev['X'],TIMES)for m in members],axis=0)).max())
                assert error<=1e-12
                detail['checks'][version]=dict(**verify_export(f,tr,ev,out/version),ensemble_mean_error=error)
        write(out/'summary.json',dict(status='COMPLETE',mode=a.mode,year=a.year,seed=a.seed,count=a.count,
            host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),seconds=time.perf_counter()-start,
            memory_slot_wait_seconds=waited,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            training_sha256=sha(root/f'folds/{a.year}/train_PRIVATE.npz'),no_evaluation_labels=True,**detail))
        write(out/'hashes.json',{str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['grow','assemble','inspect'],required=True)
    p.add_argument('--year',type=int,choices=YEARS,required=True);p.add_argument('--seed',type=int,choices=[0,*SEEDS],required=True)
    p.add_argument('--start',type=int);p.add_argument('--stop',type=int);p.add_argument('--count',type=int)
    p.add_argument('--registry');p.add_argument('--registry-sha256');a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
