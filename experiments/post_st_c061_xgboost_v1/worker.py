import sys,time,socket,resource,platform
import pandas as pd
import xgboost as xgb
from .common import *
from .pipeline import select,fit,predict

def main(y,h):
    start=time.perf_counter();inp=Path(os.environ['DATA_DIR'])/str(y);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    tr=arrays(inp/'train.npz');ev=arrays(inp/'eval.npz');assert set(ev)=={'X','sample_ids'}
    folds=[(v,arrays(inp/f'{v}_fit.npz'),arrays(inp/f'{v}_val.npz')) for v in (y-3,y-2)]
    tune=out/'tuning';tune.mkdir()
    c,rule,rows,log,nfit=select(folds,h,tune)
    write(out/'selected.json',dict(config=c,selection_rule=rule))
    pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False);pd.DataFrame(log).to_csv(out/'training_log.csv',index=False)
    pp={};final=[];train_predictions={}
    for s in SEEDS:
        model,info=fit(tr,h,c,s);p=predict(model,ev['X']);train_predictions[str(s)]=predict(model,tr['X'])
        mp=out/f'model_{s}.ubj';model.save_model(mp);copy=xgb.Booster();copy.load_model(mp)
        np.testing.assert_array_equal(p,predict(copy,ev['X']));np.testing.assert_array_equal(train_predictions[str(s)],predict(copy,tr['X']))
        pp[str(s)]=p
        tree=model.trees_to_dataframe()
        final.append(dict(seed=s,**info,reload_error=0.,model_bytes=mp.stat().st_size,trees=c['num_boost_round'],leaves=int(tree.Feature.eq('Leaf').sum()),used_features=int(tree.loc[tree.Feature!='Leaf','Feature'].nunique())))
    pp['ensemble']=np.mean(list(pp.values()),axis=0)
    np.savez_compressed(out/'predictions_PRIVATE.npz',sample_ids=ev['sample_ids'],**pp)
    np.savez_compressed(out/'training_predictions_PRIVATE.npz',sample_ids=tr['sample_ids'],**train_predictions)
    write(out/'result.json',dict(year=y,months=h,chosen=c,selection_rule=rule,tuning_fits=nfit,final_fits=len(SEEDS),xai_fits=0,final=final,seconds=time.perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,hostname=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),python=platform.python_version(),xgboost=xgb.__version__,input_hashes={p.name:sha(p) for p in inp.glob('*.npz')}))

if __name__=='__main__':main(*map(int,sys.argv[1:]))
