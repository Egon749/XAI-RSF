"""Reuse immutable manufacturing arrays and references; change no sample/feature."""
from pathlib import Path
import argparse,json,hashlib,shutil

PARENT=Path('/home/oliver/MainDiskData/xai_rsf_manufacturing_2023_2024_v1_20261027_XvjGsHee')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def main(run):
    for name in ['input','audit','reports','private']:(run/name).mkdir()
    inp=run/'input';old=json.loads((PARENT/'input/plan.json').read_text())
    protocol=Path(__file__).with_name('frozen_protocol.json');shutil.copy2(protocol,run/'audit/frozen_protocol.json')
    plan=json.loads(protocol.read_text());source={}
    for role in ['train','eval']:
        for suffix in ['_PRIVATE.npz','_metadata_PRIVATE.parquet']:
            name=f'final_{role}{suffix}';p=PARENT/'input'/name
            assert sha(p)==old['source_hashes'][name];shutil.copy2(p,inp/name);source[str(p)]=sha(p)
    for name in ['M2_PRIVATE.npz']:
        p=PARENT/'input'/name;assert sha(p)==old['source_hashes'][name];shutil.copy2(p,inp/name);source[str(p)]=sha(p)
    shutil.copytree(PARENT/'input/legacy',inp/'legacy')
    for p in (PARENT/'results/assembled').glob('*/task-*/output/summary.json'):
        s=json.loads(p.read_text());name='q2' if s['variant']=='coverage' else s['variant']
        pred=p.parent/'predictions_PRIVATE.npz';hashes=json.loads((p.parent/'output_hashes.json').read_text())
        assert sha(pred)==hashes[pred.name]
        shutil.copy2(pred,inp/f'reference_{name}_{s["seed"]}_PRIVATE.npz');source[str(pred)]=sha(pred)
        if name=='q2':
            for fname in ['forest_PRIVATE.json','functions_PRIVATE.npz']:
                src=p.parent/fname;assert sha(src)==hashes[fname]
                shutil.copy2(src,inp/f'reference_q2_{s["seed"]}_{fname}');source[str(src)]=sha(src)
    for name in ['M2_provenance.json','data_alignment.json']:
        shutil.copy2(PARENT/'audit'/name,run/'audit'/name)
    audit=json.loads((run/'audit/data_alignment.json').read_text())
    for seed in plan['seeds']:
        for q in [3,4]:
            expected=dict(old['configs'][f'final/{seed}/coverage']);expected['path_budget']=q
            assert plan['configs'][f'final/{seed}/q{q}']==expected
    plan.update(source_counts=audit['source_counts'],folds=['final'],source_hashes={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()},data_status=audit['status'])
    write(inp/'plan.json',plan)
    write(run/'audit/reuse_receipt.json',dict(parent=str(PARENT),parent_plan_sha256=sha(PARENT/'input/plan.json'),
        source_hashes=source,exact_array_copy=True,only_training_config_change='path_budget 2 -> 3 or 4',industry_PIT_certified=False,
        evaluation_status='POST_EXPOSURE_EXPLORATORY',protocol_sha256=sha(protocol)))
    print(json.dumps(audit['source_counts']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
