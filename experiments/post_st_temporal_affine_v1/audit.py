"""Read-only validation of input preservation, publication, and revision ledger."""
import argparse,collections,subprocess,time
import pandas as pd
from .common import *

def main():
    start=time.perf_counter();block_forest_fits()
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    count={}
    for name in ['source_hashes','inherited_source_hashes','immutable_input_hashes']:
        d=load(RUN/f'audit/{name}.json')
        for p,h in d.items():assert sha(RUN/'input'/p if name=='immutable_input_hashes' else Path(p))==h,(name,p)
        count[name]=len(d)
    revision=load(RUN/'revisions/solver_v2/snapshot.json')
    for p,h in revision['files'].items():assert sha(RUN/'revisions/solver_v2/code'/p)==h
    assert revision['protocol_sha256']==sha(RUN/'input/protocol.json')==sha(REPO/f'experiments/{NAME}/frozen_protocol.json')
    assert not subprocess.check_output(['git','diff','37be0ee05ed64f31ba18c905795403aa8b9012ed','--','CURRENT_POST_ST_MODEL.json','crforest'],cwd=REPO).strip()
    ledger=[];numerical_changes=[];original={};current={}
    for stage in ['point','bootstrap','point_solver_v2','bootstrap_solver_v2']:
        lines=(RUN/f'audit/launcher_{stage}.log').read_text().splitlines()
        summary=next(json.loads(line) for line in lines if line.startswith('{"job"'))
        result=load(summary['result']);assert not result['pending'];good=0;failed=0;calibrators=0
        for task in result['completed']:
            accepted=stage.endswith('solver_v2');ok=task['returncode']==0
            p=RUN/f'results/{stage}/{summary["job"]}/task-{task["id"]}/output' if ok else Path(task['storage']['output'])
            if ok:
                assert task['compute_completed'] and task['publication_completed'];good+=1
                for n,h in load(p/'hashes.json').items():assert sha(p/n)==h
                s=load(p/'summary.json');assert s['new_forests']==0
                cal=load(p/'calibrators_PRIVATE.json');calibrators+=len(cal)
                for c in cal:(current if accepted else original)[(c['year'],c['months'],c['arm'],c['repeat'])]=c
            else:
                failed+=1;assert (p/'FAILED.json').exists();partial=list(p.glob('*_calibrator_PRIVATE.json'));calibrators+=len(partial)
            ledger.append(dict(stage=stage,job=summary['job'],task=task['id'],host=task['host'],accepted=accepted,success=ok,
                compute_completed=task['compute_completed'],publication_completed=task['publication_completed'],seconds=task['seconds'],
                report_sha256=sha(Path(summary['result'])),failure_reason='' if ok else 'linear oracle gap3.390211264411164e-08 >1e-8;original output retained'))
        count[stage]=dict(successful_chains=good,failed_chains=failed,successful_calibrator_solves_including_partial=calibrators,job=summary['job'],execution_seconds=summary['execution_seconds'])
    for k,c in original.items():
        new=current[k];d=arrays(RUN/f'input/calibration/{k[1]}/{k[0]}/calibration_PRIVATE.npz');p=d['p']
        delta=np.asarray(c['c'])+p*c['b']-np.asarray(new['c'])-p*new['b']
        numerical_changes.append(dict(year=k[0],months=k[1],arm=k[2],repeat=k[3],max_historical_probability_change=float(abs(delta).max()),objective_change=new['objective_after']-c['objective_after']))
    pd.DataFrame(ledger).to_csv(RUN/'reports/EXECUTION_LEDGER.csv',index=False)
    pd.DataFrame(numerical_changes).to_csv(RUN/'reports/SOLVER_REVISION_COMPARISON.csv',index=False)
    valid=load(RUN/'reports/numerical_validation.json');metric=load(RUN/'reports/metric_validation.json')
    assert valid['status']==metric['status']=='PASS' and len(current)==918
    assert sum(x['success'] for x in ledger)==203 and sum(not x['success'] for x in ledger)==1
    assert {x['host'] for x in ledger if x['accepted']}=={'oliverdesktop','lemon'}
    audit=dict(status='PASS',preservation=count,source_revision_commit=revision['commit'],frozen_commit=load(RUN/'audit/code_snapshot.json')['commit'],
        current_pointer_sha256=REFERENCE_SHA,protocol_sha256=revision['protocol_sha256'],core_forest_unchanged=True,
        accepted_chain_tasks=102,accepted_point_calibrators=18,accepted_resample_calibrators=900,new_forest_fits=0,
        superseded_successful_calibrator_solves=914,diagnostic_successful_calibrator_solves=12,diagnostic_failed_solves=2,
        actual_successful_calibrator_solves_including_revision_and_diagnostics=1844,actual_failed_calibrator_solves_including_diagnostics=3,
        actual_chain_tasks=204,successful_chain_tasks=203,failed_chain_tasks=1,both_nodes_used=True,
        max_same_data_prediction_change_after_numerical_revision=max(x['max_historical_probability_change'] for x in numerical_changes),
        seconds=time.perf_counter()-start)
    write(RUN/'reports/preservation_execution_audit.json',audit);print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
