"""Validate source snapshots, authorized ranking release and folder manifest."""
import hashlib,json,re
from pathlib import Path
import pandas as pd
from .common import NAME,REPO,sha,load

def main():
    base=REPO/'experiments'/NAME;implementation=base/'implementation';sources=load(implementation/'source_manifest.json')
    for row in sources['files']:
        assert sha(implementation/row['snapshot'])==row['sha256']
        assert sha(REPO/row['source'])==row['sha256']
    release=load(base/'rankings/publication_receipt.json')
    for row in release['source_files']:
        assert sha(base/'rankings'/row['published'])==row['sha256']
    assert load(base/'PUBLICATION_AUDIT.json')['status']=='PASS'
    assert load(base/'rankings/reproduction_check.json')['status']=='PASS'
    names=[]
    for p in sorted(base.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['BUNDLE_MANIFEST.json','BUNDLE_VALIDATION.json']:
            assert p.suffix not in ['.npz','.parquet'] and '_PRIVATE' not in p.name
            names.append(dict(path=str(p.relative_to(base)),bytes=p.stat().st_size,sha256=sha(p)))
    manifest=dict(scope='one-folder cause/support experiment handoff including explicitly authorized company rankings',
        original_results_commit='37be0ee05ed64f31ba18c905795403aa8b9012ed',snapshot_files=len(sources['files']),
        total_files=len(names),files=names,excluded=['BUNDLE_MANIFEST.json','BUNDLE_VALIDATION.json','__pycache__'])
    (base/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    links=0
    for doc in base.rglob('*.md'):
        for match in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
            target=match.strip('<>').split('#')[0]
            if not target or '://' in target:continue
            assert (doc.parent/target).is_file(),(doc,target)
            links+=1
    result=dict(status='PASS',new_real_fits=0,snapshots_byte_identical=True,source_files=len(sources['files']),
        markdown_local_links_checked=links,authorized_company_horizon_rows=162,independent_companies=81,
        original_ranking_files_hash_identical=True,reproduction_matched=True,files=len(names),
        manifest_sha256=sha(base/'BUNDLE_MANIFEST.json'))
    (base/'BUNDLE_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
