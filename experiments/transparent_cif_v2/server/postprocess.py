"""Verify published tasks; evaluate frozen predictions; prepare deployment/bootstrap."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
import numpy as np
import pandas as pd
from crforest.transparent import file_sha256 as sha,write_json

def gate(run,job):
    report=json.loads((run/f'formal.jsonl.results-{job}.json').read_text())
    assert report['expected']==4 and len(report['completed'])==4 and not report['pending']
    assert {r['id'] for r in report['completed']}==set(range(4))
    assert all(r['returncode']==0 and r['compute_completed'] and r['publication_completed'] for r in report['completed'])
    outputs=[];rows=[]
    for task in range(4):
        root=run/'results/formal'/str(job)/f'task-{task}'
        rec=json.loads((root/'complete.json').read_text());out=root/'output'
        for entry in rec['manifest']:
            if entry[1]=='directory':continue
            name,size,digest=entry;assert (out/name).stat().st_size==size and sha(out/name)==digest
        summary=json.loads((out/'summary.json').read_text());assert summary['status']=='COMPLETE'
        assert summary['seed']==([42,101,202,-1][task]);rows.append(dict(task=task,**summary,publication_sha256=sha(root/'complete.json')));outputs.append(out)
    write_json(run/'audit/publication_gate.json',dict(status='PASS',job=job,tasks=rows))
    return outputs

def main(a):
    run=a.run;repo=Path(__file__).resolve().parents[3];src=run/'input';outputs=gate(run,a.job)
    predictiondirs=[p/'predictions' for p in outputs[:3]];evaluation=run/'evaluation'
    subprocess.run([sys.executable,'-m','experiments.transparent_cif_v2.evaluate','--source-input',str(src),'--predictions',*[str(p) for p in predictiondirs],'--output',str(evaluation)],check=True)
    d=pd.read_parquet(src/'final_eval_metadata_PRIVATE.parquet');pred={}
    for out in outputs[:3]:
        for it in json.loads((out/'predictions/prediction_receipt.json').read_text())['models']:
            with np.load(out/'predictions'/f'{it["model"]}_PRIVATE.npz') as z:pred[f'{it["model"]}|{it["seed"]}']=z['risk']
    with np.load(src/'M2_PRIVATE.npz') as z:
        for seed,p in zip(z['seeds'],z['probabilities']):pred[f'M2_allsector_reference|{seed}']=p
    for seed in [42,101,202]:
        with np.load(src/f'reference_q2_{seed}_PRIVATE.npz') as z:pred[f'q2_fulltrain_reference|{seed}']=z['cif12'][:,0]
    for model in sorted({k.split('|')[0] for k in pred}):pred[model+'|-1']=np.mean([v for k,v in pred.items() if k.split('|')[0]==model],axis=0)
    # Original company-bootstrap helpers and immutable predictions on shared NFS.
    bs=run/'bootstrap_input';bs.mkdir();d.to_parquet(bs/'metadata.parquet',index=False);np.savez_compressed(bs/'predictions.npz',**pred);shutil.copytree(src/'legacy',bs/'legacy')
    shutil.copy2(repo/'experiments/transparent_cif_v2/server/bootstrap.py',bs/'worker.py')
    launch="""import os,socket,sys\nfrom pathlib import Path\np='/home/oliver/.venvs/xai-baselines-20261027/bin/python' if socket.gethostname().startswith('oliver') else '/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'\nfor k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'\nos.execv(p,[p,str(Path(__file__).with_name('worker.py')),*sys.argv[1:]])\n"""
    (bs/'launch.py').write_text(launch)
    manifest=run/'bootstrap.jsonl';tasks=[dict(kind='cpu',argv=['python3',str(bs/'launch.py'),'--batch',str(i)],timeout=7200,storage=dict(source=str(bs),mode='nfs',results=str(run/'results/bootstrap'))) for i in range(10)]
    with manifest.open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    write_json(run/'audit/bootstrap_freeze.json',dict(replicates=1000,seed=20261025,manifest_sha256=sha(manifest),source_hashes={str(p.relative_to(bs)):sha(p) for p in bs.rglob('*') if p.is_file()},same_company_draws_across_models_scopes_and_seeds=True,models=len(pred),fits=0))
    sys.path.insert(0,str(src/'legacy'))
    from evaluate import top_flags,warning_metrics,lead_summary,event_leads
    from common import endpoint_weights
    from endpoint_guard import source_consistent_endpoint_weights
    warning=[];cal=[]
    scopes=[('pooled',np.arange(len(d)))]+[(str(y),np.asarray(ii)) for y,ii in d.groupby('year').indices.items()]+[(str(q),np.asarray(ii)) for q,ii in d.groupby('quarter').indices.items()]
    for scope,ix in scopes:
        dd=d.iloc[ix].reset_index(drop=True)
        for key,allp in pred.items():
            model,seed=key.split('|');p=allp[ix]
            for frac in [.2,.25]:
                flags=top_flags(dd,p,frac);wm=warning_metrics(dd,flags);hit=flags&(dd.event.to_numpy()==1)
                unknown=flags&~(dd.label_eligible.to_numpy(bool)|(dd.event.to_numpy()==2))
                ag=set(dd.loc[flags,'group']);hg=set(dd.loc[hit,'group']);ug=set(dd.loc[unknown,'group'])-hg
                assert len(ag)==len(hg)+len(ug)+len(ag-hg-ug)
                warning.append(dict(scope=scope,model=model,seed=int(seed),top_fraction=frac,**wm,**lead_summary(event_leads(dd,flags)),hit_company_groups=len(hg),no_hit_with_unknown_groups=len(ug),no_hit_all_alerts_known_non_target_groups=len(ag-hg-ug),company_hit_fraction=len(hg)/len(ag)))
            if scope in ['2023','2024','pooled']:
                w,_=source_consistent_endpoint_weights(dd,endpoint_weights)
                order=np.lexsort((dd.sample_id.to_numpy(),p))
                for b,ii in enumerate(np.array_split(order,10)):
                    cal.append(dict(scope=scope,model=model,seed=int(seed),bin=b+1,records=len(ii),prediction_mean=float(p[ii].mean()),ipcw_observed_incidence=float(np.sum(w[ii]*(dd.event.to_numpy()[ii]==1))/len(ii)),unknown_records=int(np.sum((dd.event.to_numpy()[ii]==0)&~dd.label_eligible.to_numpy(bool)[ii]))))
    pd.DataFrame(warning).to_csv(evaluation/'warning_company_quarter.csv',index=False);pd.DataFrame(cal).to_csv(evaluation/'calibration.csv',index=False)
    # A standalone evaluator sees neither original forest code nor tree banks.
    deploy=run/'standalone_deployment';deploy.mkdir();(deploy/'crforest').mkdir();(deploy/'crforest/__init__.py').write_text('')
    for name in ['functions.py','transparent.py']:shutil.copy2(src/'code/crforest'/name,deploy/'crforest'/name)
    shutil.copy2(repo/'experiments/transparent_cif_v2/server/deploy_verify.py',deploy/'verify.py')
    with np.load(src/'final_eval_PRIVATE.npz') as z:np.savez_compressed(deploy/'inputs_PRIVATE.npz',X=z['X'],horizon=z['horizon'],sample_ids=z['sample_ids'])
    index=[]
    for out in outputs[:3]:
        for it in json.loads((out/'fit/model_index.json').read_text()):
            if it.get('kind')=='endpoint_ebm':continue
            name=f'{it["name"]}_{it["seed"]}';folder=deploy/name;folder.mkdir()
            for fn in ['functions_PRIVATE.npz','manifest_PRIVATE.json','integrity.json']:shutil.copy2(out/'fit'/it['path']/fn,folder/fn)
            with np.load(out/'predictions'/f'{it["name"]}_PRIVATE.npz') as z:np.savez_compressed(folder/'expected_PRIVATE.npz',cif=z['cif12'][:,None,:],sample_ids=z['sample_ids'])
            index.append(dict(model=it['name'],seed=it['seed'],path=name))
    write_json(deploy/'index.json',index)
    with (run/'audit/standalone_deployment.log').open('x') as f:subprocess.run([sys.executable,'-I',str(deploy/'verify.py')],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=deploy)
    write_json(run/'audit/standalone_runtime_hashes.json',{str(p.relative_to(deploy)):sha(p) for p in deploy.rglob('*.py')})
    print('Verified 4 published tasks; evaluated 40 prediction sets; prepared 10 bootstrap tasks; independently deployed 18 native models.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--job',type=int,required=True);main(p.parse_args())
