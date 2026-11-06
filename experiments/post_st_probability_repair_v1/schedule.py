import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from experiments.post_st_benchmark_calibration_v1.common import sha,write

def main():
    repo=Path(__file__).resolve().parents[2]
    root=Path(os.environ['REPAIR_RUN'])
    inp=root/'input'; code=inp/'code';code.mkdir()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    archive=subprocess.check_output(['git','archive',rev,'--','crforest','experiments'],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:t.extractall(code,filter='data')
    items=[]
    for year in [2023,2024,2025]:
        items.append(dict(kind='cpu',argv=['python3',str(code/'experiments/post_st_probability_repair_v1/launch.py'),str(year)],
                          cwd=str(code),timeout=3600,storage=dict(source=str(inp),mode='nfs',results=str(root/'published'))))
    manifest=root/'tasks.jsonl';manifest.write_text(''.join(json.dumps(i)+'\n' for i in items))
    write(root/'snapshot.json',dict(worker_commit=rev,manifest_sha256=sha(manifest)))
    subprocess.run([str(Path.home()/'.local/bin/hetero-run'),str(manifest)],check=True)

if __name__=='__main__':main()
