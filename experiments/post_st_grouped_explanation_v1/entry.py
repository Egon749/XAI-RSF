import importlib,socket,sys,time
from .common import *
stage=sys.argv[1];year,first,last=map(int,sys.argv[2:])
out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True);start=time.perf_counter()
importlib.import_module('.'+stage,__package__).worker(year,first,last,out,Path(os.environ['DATA_DIR']))
dump(out/'done.json',dict(args=sys.argv[1:],host=socket.gethostname(),seconds=time.perf_counter()-start))
