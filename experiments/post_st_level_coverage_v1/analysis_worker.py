"""Disjoint saved-output analysis tasks; never fit a forest."""
from pathlib import Path
import argparse,os,resource,time,traceback
from .common import sha,write

def main(run,scenario):
    out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    start=time.perf_counter()
    try:
        if scenario:
            from .sim_analysis import analyze_simulation
            analyze_simulation(run,scenario,out)
        else:
            from .analyze import enterprise
            enterprise(run,out)
        write(out/'summary.json',dict(status='COMPLETE',scenario=scenario,analysis_only=True,
            seconds=time.perf_counter()-start,maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='hashes.json'})
    except Exception:
        write(out/'FAILED.json',dict(status='FAILED',traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--scenario');a=p.parse_args();main(a.run,a.scenario)
