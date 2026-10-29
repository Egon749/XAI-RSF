"""One Slurm CPU task using the existing pool storage publication protocol.
The installed coordinator hardcodes oliver's address; this permits a lemon-only
allocation without changing the installed scheduler or pool implementation.
"""
import os,sys,json,socket,time,subprocess,traceback
from pathlib import Path
sys.path.insert(0,'/home/oliver/.local/share/heteropool-20261005')
import storage
p=Path(sys.argv[1]);tasks=[json.loads(x) for x in p.read_text().splitlines()];assert len(tasks)==1;t=tasks[0];job=os.environ['SLURM_JOB_ID'];start=time.monotonic();io=None;rc=70;compute_rc=None;phase='prepare';log=Path(str(p)+'.log-'+job)
try:
 storage.plan(tasks);storage.preflight([t['storage']],job,create=True);io=storage.prepare(t['storage'],job,0);env=dict(os.environ,DATA_DIR=io['data'],OUTPUT_DIR=io['output'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
 phase='compute'
 with log.open('x') as f:r=subprocess.run(t['argv'],cwd=t['cwd'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=t['timeout'])
 rc=compute_rc=r.returncode
 if rc==0:
  phase='publish';storage.record_computed(t['storage'],io,job,0,t);storage.publish(t['storage'],io,job,0);phase='complete'
except Exception:
 with log.open('a') as f:f.write(traceback.format_exc())
 rc=75 if phase=='publish' else 70
item=dict(id=0,host=socket.gethostname(),returncode=rc,compute_returncode=compute_rc,compute_completed=compute_rc==0,publication_completed=bool(io and io.get('publication_state')=='verified'),phase=phase,storage=io,seconds=time.monotonic()-start,log=str(log))
Path(str(p)+'.results-'+job+'.json').write_text(json.dumps(dict(expected=1,completed=[item],pending=[],seconds=time.monotonic()-start,execution='single Slurm task with unchanged storage protocol'),indent=2));sys.exit(rc)
