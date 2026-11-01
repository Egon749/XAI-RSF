"""Publication-only recovery; preserve the pool's identity/hash/lock protocol.

The original NFS rsync hit its fixed 600-second limit on many small files.
Run on the ORIGINAL compute node. Only the copy transport is replaced by an
authoritative-server pull; the original storage.publish performs every source
and destination hash check and the no-overwrite commit. No fitting entry point
is imported or executed.
"""
from pathlib import Path
import argparse,contextlib,datetime,hashlib,importlib.util,io,json,shlex,socket,subprocess,time

STORAGE=Path('/home/oliver/.local/share/heteropool-20261005/storage.py')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def relay_command(argv,receipt):
    if len(argv)!=5 or list(argv[:3])!=['rsync','-rlt','--']:
        raise ValueError('Unexpected publication copy command')
    source=Path(argv[3]);destination=Path(argv[4])
    if source!=Path(receipt['info']['output']):raise ValueError('Source differs from receipt')
    root=Path(receipt['spec']['results'])/str(receipt['job'])
    relative=destination.relative_to(root)
    if len(relative.parts)!=2 or not relative.parts[0].startswith(f'.partial-{receipt["task"]}-') or relative.parts[1]!='output':
        raise ValueError('Destination is not the pool-owned staging directory')
    # shlex.join is shell quoting, not JSON/string interpolation. rsync -s also
    # protects the remote source path. Final task-* paths are never written here.
    pull=['rsync','-rlt','-s','--',f'lemon:{source}/',str(destination)+'/']
    return ['ssh','-o','BatchMode=yes','oliver',shlex.join(pull)]

def resume(receipt_path,audit):
    receipt_path=Path(receipt_path);receipt=json.loads(receipt_path.read_text())
    if receipt['host']!=socket.gethostname() or receipt['host']!='lemon':raise ValueError('Resume on original lemon node')
    if receipt['compute_returncode']!=0 or receipt['state']!='publication_failed':raise ValueError('Only failed publication of completed compute may recover')
    audit.mkdir(parents=True,exist_ok=True);task=int(receipt['task'])
    backup=audit/f'original_publication_{task}.json'
    with backup.open('xb') as f:f.write(receipt_path.read_bytes())
    spec=importlib.util.spec_from_file_location('h24_pool_storage_recovery',STORAGE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_run=module.subprocess.run
    def run(argv,**kwargs):
        if argv and argv[0]=='rsync':
            argv=relay_command(argv,receipt);kwargs['timeout']=3600
        return original_run(argv,**kwargs)
    result=dict(job=receipt['job'],task=task,recovery_only=True,fitting_executed=False,
                transport='authoritative_server_pull_rsync',original_receipt_sha256=digest(backup),
                storage_source_sha256=digest(STORAGE),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    stream=io.StringIO();start=time.monotonic();module.subprocess.run=run
    try:
        with contextlib.redirect_stdout(stream):module.resume(str(receipt_path))
        result.update(status='VERIFIED',original_protocol_output=json.loads(stream.getvalue().strip()))
    except Exception as exc:
        result.update(status='FAILED',error=repr(exc));raise
    finally:
        module.subprocess.run=original_run;result['seconds']=time.monotonic()-start
        with (audit/f'recovery_{task}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

def reconcile(run):
    # Preserve the original failed job report; build a separate recovered gate.
    from .fitting import load,sha,write
    launcher=run/'audit/launcher_bootstrap.log';receipt=None
    for line in launcher.read_text().splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and 'job' in value and 'result' in value:receipt=value
    if receipt is None:raise ValueError('Original launcher has not finished')
    report=load(Path(receipt['result']));items=load(run/'input/bootstrap/plan.json')['items']
    assert not report['pending'] and report['expected']==len(items)
    assert {int(s['id']) for s in report['completed']}==set(range(len(items)))
    records=[];failures=[]
    for state in report['completed']:
        i=int(state['id']);assert state['compute_completed'] and state['compute_returncode']==0
        recovered=not state['publication_completed']
        out=run/f'results/bootstrap/{receipt["job"]}/task-{i}/output';marker=load(out.parent/'complete.json')
        assert str(marker['job'])==str(receipt['job']) and marker['task']==i
        if recovered:
            recovery=load(run/f'private/publication_recovery/recovery_{i}.json');old=load(run/f'private/publication_recovery/original_publication_{i}.json')
            assert recovery['status']=='VERIFIED' and recovery['fitting_executed'] is False
            assert marker['publication_id']==old['info']['publication_id'] and marker['manifest']==old['manifest']
            failures.append(dict(task=i,phase=state.get('phase'),returncode=state['returncode'],compute_returncode=state['compute_returncode'],
                reason=old['last_error'],recovery_seconds=recovery['seconds'],recovery_receipt_sha256=sha(run/f'private/publication_recovery/recovery_{i}.json')))
        else:assert state['returncode']==0
        for name,h in load(out/'hashes.json').items():assert sha(out/name)==h
        summary=load(out/'summary.json');assert summary['status']=='COMPLETE'
        records.append(dict(**items[i],task_id=i,output=str(out),new_forests=summary['new_forests'],host=summary['host'],
                            initial_publication_complete=not recovered,publication_recovered=recovered))
    write(run/'audit/publication_recovery_summary.json',dict(original_job=receipt['job'],original_job_success=receipt['success'],
        original_report=receipt['result'],original_report_sha256=sha(Path(receipt['result'])),
        original_publication_failures=len(failures),verified_recoveries=len(failures),additional_fits=0,failures=failures,
        transport='authoritative-server rsync pull inside unchanged pool publication identity/hash/lock/no-overwrite protocol'))
    write(run/'audit/registry_bootstrap.json',records)
    write(run/'audit/gate_bootstrap.json',dict(status='PASS',stage='bootstrap',job=receipt['job'],tasks=len(items),
        new_forests=sum(r['new_forests'] for r in records),seconds=receipt['execution_seconds'],report=receipt['result'],
        report_sha256=sha(Path(receipt['result'])),publication_recovery_required=True,original_job_success=receipt['success']))
    print(json.dumps(dict(status='PASS_AFTER_PUBLICATION_ONLY_RECOVERY',tasks=len(items),recovered=len(failures),additional_fits=0)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path);p.add_argument('--audit',type=Path);p.add_argument('--reconcile',type=Path);a=p.parse_args()
    if a.reconcile:reconcile(a.reconcile)
    else:
        if a.receipt is None or a.audit is None:p.error('--receipt and --audit required')
        resume(a.receipt,a.audit)
