"""Three raw fits per paired outer draw, followed by 0/10/18 derivatives."""
import argparse,os,resource,socket,time,traceback
from crforest import CompetingRiskForest,CIFExpansion,ForestConfig
from crforest.shrinkage import shrink_leaf_cifs
from experiments.post_st_level_coverage_v1.worker import verify_export
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_shrinkage_strength_v1.verification import check_shrinkage
from experiments.post_st_ablation_v1.common import pairing,array_hash
from .common import *

def execute(year,repeat,inp,out):
    original=arrays(inp/f'folds/{year}/train_PRIVATE.npz')
    ev=arrays(inp/f'folds/{year}/eval_PRIVATE.npz')
    assert set(ev)=={'X','sample_ids'} and original['X'].shape[1]==74
    tr,ix=resample(original,year,repeat)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'draw_PRIVATE.npz',indices=ix,multiplicity=np.bincount(ix,minlength=len(ix)))
    members={s:[] for s in STRENGTHS};rows=[];pred={s:[] for s in STRENGTHS}
    for seed in SEEDS:
        cfg=ForestConfig(**load(inp/'configs.json')[str(seed)])
        start=time.perf_counter()
        f=CompetingRiskForest(cfg).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1)
        elapsed=time.perf_counter()-start
        pa=pairing(tr['groups'],cfg,74)
        for actual,expected in zip(f.inbag_,pa):assert array_hash(actual)==expected['multiplicity_sha256']
        folder=out/str(seed);folder.mkdir()
        write(folder/'pairing_PRIVATE.json',pa)
        for strength in STRENGTHS:
            begin=time.perf_counter()
            sh=f if strength==0 else shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],strength)
            shrink_seconds=time.perf_counter()-begin
            replay=check_shrinkage(f,sh,tr,strength)
            # ORIGINAL complete training cohort is the common explanation reference.
            check=verify_export(sh,original,ev,folder/str(strength))
            assert check['max_path_variables']<=2
            leaves=[n for t in sh.trees_ for n in t if 'cif' in n]
            rows.append(dict(seed=seed,strength=strength,fit_seconds=elapsed if strength==0 else 0.,
              shrink_seconds=shrink_seconds,checks=check,independent=replay,
              mean_unique_leaf_companies=float(np.mean([n['groups'] for n in leaves])),
              singleton_target_leaves=sum(n['cause_company_support'][0]==1 for n in leaves)))
            members[strength].append(sh);pred[strength].append(sh.predict(ev['X'],TIMES))
    ensembles=[]
    for strength in STRENGTHS:
        f=merge(members[strength]);folder=out/f'ensemble_{strength}'
        ck=verify_export(f,original,ev,folder)
        expected=np.mean(pred[strength],axis=0)
        err=float(abs(expected-f.predict(ev['X'],TIMES)).max());assert err<1e-12
        np.savez_compressed(folder/'mean_predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],times=TIMES,cif=expected)
        # Local explanations on a fixed evaluation population, with fixed training centering.
        ex=CIFExpansion.load(folder/'functions_PRIVATE.npz');keys=[(j,) for j in ex.mains]+ex.pairs
        tt=np.stack([ex.component(ev['X'],k,TIMES) for k in keys],axis=1) if keys else np.empty((len(ev['X']),0,4,2))
        np.savez_compressed(folder/'local_components_PRIVATE.npz',keys=np.array([','.join(map(str,k)) for k in keys]),terms=tt,mu=ex._coeff(TIMES)[1])
        ensembles.append(dict(strength=strength,checks=ck,mean_error=err))
    u=np.unique(ix)
    return dict(status='COMPLETE',year=year,repeat=repeat,forest_fits=3,
      outer_rows=len(ix),unique_training_companies=len(u),
      distinct_termination=int((original['event'][u]==1).sum()),
      distinct_recovery=int((original['event'][u]==2).sum()),
      weighted_termination=int((tr['event']==1).sum()),weighted_recovery=int((tr['event']==2).sum()),
      draw_sha256=array_hash(ix),members=rows,ensembles=ensembles,
      source_train_sha256=sha(inp/f'folds/{year}/train_PRIVATE.npz'),no_evaluation_labels_read=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('year',type=int,choices=YEARS);ap.add_argument('repeat',type=int);a=ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    inp=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);start=time.perf_counter()
    try:
        result=execute(a.year,a.repeat,inp,out)
        result.update(host=socket.gethostname(),slurm_job=os.environ.get('SLURM_JOB_ID'),
          seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        write(out/'summary.json',result)
        write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
        print(json.dumps({k:v for k,v in result.items() if k not in ('members','ensembles')}),flush=True)
    except Exception:
        out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()

