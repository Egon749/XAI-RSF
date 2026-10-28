"""Verify and copy frozen manufacturing inputs; no rebuilding or new dates."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tarfile
import numpy as np
from experiments.transparent_cif_v2.run import training_contract,feature_contract
from crforest.transparent import group_roles


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def prepare(run,parent):
    inp=run/'input';src=parent/'input';expected=json.loads((parent/'audit/input_hashes.json').read_text());provenance={}
    names=['final_train_PRIVATE.npz','final_eval_PRIVATE.npz','final_train_metadata_PRIVATE.parquet','final_eval_metadata_PRIVATE.parquet','plan.json','M2_PRIVATE.npz','references_PRIVATE.npz',*[f'reference_q2_{s}_PRIVATE.npz' for s in [42,101,202]]]
    for name in names:
        assert sha(src/name)==expected[name];shutil.copy2(src/name,inp/name);assert sha(inp/name)==expected[name];provenance[str(src/name)]=expected[name]
    (inp/'legacy').mkdir()
    for p in (src/'legacy').glob('*.py'):
        assert sha(p)==expected['legacy/'+p.name];shutil.copy2(p,inp/'legacy'/p.name);provenance[str(p)]=sha(p)
    old=Path('/home/oliver/LLMworkspace/early-warning-12m-model-comparison-2012plus/experiments/early_warning_12m/bootstrap.py')
    shutil.copy2(old,inp/'legacy/bootstrap.py');provenance[str(old)]=sha(old)
    base=Path('/home/oliver/MainDiskData/newrsf2_real_v1_20261026/input')
    content=json.loads((base/'features.json').read_text());quality=json.loads((base/'quality_features.json').read_text())
    matrixpath=Path('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025/input/model_matrix.json')
    matrix=json.loads(matrixpath.read_text());assert [x['name'] for x in content]==matrix['features'] and len(content)==72 and len(quality)==3
    assert [x['name'] for x in quality]==['financial_period_age','financial_disclosure_age','market_missing_coverage']
    for p in [base/'features.json',base/'quality_features.json',matrixpath]:provenance[str(p)]=sha(p)
    fields=content+quality
    for f in fields:
        if 'UNKNOWN' in f['unit']:f['unit']='UNVERIFIED; '+f['unit']
    write(inp/'features.json',dict(features=fields,source='unchanged inherited 72 raw coordinates plus 3 quality coordinates; unknown units remain UNVERIFIED'))
    a,d,h,eligible=training_contract(inp,'2022-01-01');feature_contract(inp/'features.json',75)
    st,wt=group_roles(a['groups'],.2,20261028)
    counts={}
    for name,ix in [('all',np.arange(len(d))),('structure',st),('internal_weight_selection',wt)]:
        dd=d.iloc[ix];counts[name]=dict(rows=len(dd),groups=int(dd.group.nunique()),target_rows=int((dd.event==1).sum()),target_events=int(dd.loc[dd.event==1,'event_id'].nunique()),competing_rows=int((dd.event==2).sum()),competing_events=int(dd.loc[dd.event==2,'event_id'].nunique()),source_unknown_endpoint_rows=int((~eligible[ix,0]).sum()))
    write(run/'audit/input_alignment.json',dict(status='PROVISIONAL',source_hashes=provenance,feature_order_validated=True,feature_count=75,unknown_units=sum('UNVERIFIED' in f['unit'] for f in fields),counts=counts,group_overlap=len(set(a['groups'][st])&set(a['groups'][wt])),calendar_horizon_days=sorted(set(h[:,0].tolist())),fit_cutoff='2022-01-01',original_samples_rebuilt=False,availability_dates_fabricated=False,limitations=['Financial raw-version PIT is not certified','Industry classification timing inherits prior provisional audit','Unknown labels not set to normal']))
    print(json.dumps(counts))
def freeze(run,repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    code=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',code],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as f:f.extractall(run/'input/code',filter='data')
    args=[['fit_predict','--seed',str(s)] for s in [42,101,202]]+[['demo']]
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments/transparent_cif_v2/server/launch.py'),*a],cwd=str(run/'input/code'),timeout=21600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal'))) for a in args]
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'formal.py').write_text('# Run this independent seed manifest through hetero-run.\n');write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/freeze.json',dict(authoritative_commit='fd4a87aee54c70d5e4d165b1e7fa7011fc43c5db',execution_commit=code,manifest_sha256=sha(run/'formal.jsonl'),code_archive_sha256=sha(run/'audit/code.tar'),input_hashes={str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()},bootstrap=dict(seed=20261025,replicates=1000,predictions='fixed',alert_lists='fixed, no reranking',nuisance='fixed source-consistent legacy IPCW',scope='conditional evaluation uncertainty, not refitting uncertainty'),resource_policy='One CPU per independent seed plus synthetic demo; serial pair search preserved; not all-node utilization claimed'))
    print(json.dumps(dict(code=code,tasks=4)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--repo',type=Path);p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze(a.run,a.repo) if a.freeze else prepare(a.run,a.parent)
