from pathlib import Path
import argparse,hashlib,json,os,resource,socket,subprocess,sys,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def main(a):
    start=time.perf_counter();resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));src=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR'])
    env=os.environ.copy()
    def execute(module,args,label):
        with (out/f'{label}.log').open('x') as f:subprocess.run([sys.executable,'-m',module,*map(str,args)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    if a.stage=='fit_predict':
        execute('experiments.transparent_cif_v2.run',['train','--source-input',src,'--features',src/'features.json','--seed',a.seed,'--tree-workers','1','--ebm-control','--output',out/'fit'],'train')
        execute('experiments.transparent_cif_v2.run',['predict','--source-input',src,'--trained',out/'fit','--output',out/'predictions'],'predict')
    elif a.stage=='demo':execute('experiments.transparent_cif_v2.demo',['--output',out/'demo'],'demo')
    else:raise ValueError(a.stage)
    write(out/'summary.json',dict(stage=a.stage,seed=a.seed,status='COMPLETE',seconds=time.perf_counter()-start,peak_children_rss_MiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,host=socket.gethostname(),affinity_count=len(os.sched_getaffinity(0))))
    write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});print('Completed',a.stage,a.seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit_predict','demo']);p.add_argument('--seed',type=int,default=-1);a=p.parse_args()
    try:main(a)
    except Exception:
        write(Path(os.environ['OUTPUT_DIR'])/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise
