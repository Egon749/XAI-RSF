"""Hash-verified historical reuse; new fits never overwrite prior bundles."""
from pathlib import Path
from experiments.post_st_dropcolumn_selection_v1.common import load,sha,subset_hash
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import read_bundle,fit_bundle as ordinary_fit
from .fitting import fit_bundle
from .selection import FULL
V2=Path('/home/oliver/MainDiskData/xai_rsf_group_selection_v2_20261101_G4bKZ12C')


def previous_output(year,repeat):
    records=load(V2/'audit/registry_pilot_recovered.json')+load(V2/'audit/registry_batch.json')
    r=next(r for r in records if r['year']==year and r['repeat']==repeat)
    return Path(r['output'])


class Store:
    def __init__(self,parent,out,draw,groups,year,repeat,mode='anchored',baseline=None):
        self.parent,self.out,self.draw,self.groups=parent,out,draw,groups
        self.mode,self.baseline=mode,baseline
        self.previous=previous_output(2025 if repeat<0 else year,repeat)
        self.cache={};self.paths={};self.records=[];self.new_forests=0
    def path(self,keep,year):
        self.get(keep,year);return self.paths[(tuple(keep),year)]
    def get(self,keep,year):
        keep=tuple(keep);key=(keep,year)
        if key in self.cache:return self.cache[key]
        name=f'{year}_{subset_hash(keep)[:16]}';old=self.previous/'discovery'/name
        reused=(keep==FULL or self.mode=='uncontrolled') and (old/'hashes.json').exists()
        if reused:
            for rel,h in load(old/'hashes.json').items():assert sha(old/rel)==h
            s=load(old/'summary.json');assert tuple(s['keep'])==keep and s['year']==year
            p=old;value=read_bundle(p)
        else:
            p=self.out/name
            if self.mode=='uncontrolled':value=ordinary_fit(self.parent,keep,year,p,self.draw)
            else:
                base=None if keep==FULL else self.path(FULL,year)
                value=fit_bundle(self.parent,keep,year,p,self.draw,groups_spec=self.groups,baseline_path=base)
            self.new_forests+=3
        self.paths[key]=p;self.cache[key]=value
        self.records.append(dict(cohort=year,keep=list(keep),path=str(p),reused=reused,mode=self.mode,summary_sha256=sha(p/'summary.json'),hash_manifest_sha256=sha(p/'hashes.json')))
        return value
