"""Read local CSMAR exports without changing their contents.

Each task owns one output directory. XLSX is streamed, including workbooks
whose stored dimensions incorrectly report a single column.
"""
from pathlib import Path
import argparse, hashlib, json, os, resource, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MARKET = ['Stkcd','Trddt','Clsprc','Hiprc','Loprc','Dnshrtrd','Dnvaltrd',
          'Dsmvosd','Dsmvtll','Dretwd','Markettype','Trdsta']
INDUSTRY = ['Stkcd','Accper','Typrep','Indcd','Indnme','Indcd1','Indnme1']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def normalize(d, kind):
    d['Stkcd']=d.Stkcd.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(6)
    d=d[d.Stkcd.str.fullmatch(r'\d{6}')].copy()
    date='Trddt' if kind=='market' else 'Accper'
    d[date]=pd.to_datetime(d[date],errors='coerce')
    d=d[d[date].notna()].copy()
    if kind=='market':
        for c in MARKET[2:]:d[c]=pd.to_numeric(d[c],errors='coerce')
        d=d[d.Markettype.isin([1,4,16,32,64])]
        # All newer CSV rows are retained; overlapping vintages are reconciled
        # explicitly by the assembler, never silently used twice.
        d=d[d.Trddt.between('2010-01-01','2026-08-31')]
    else:
        for c in INDUSTRY[2:]:d[c]=d[c].fillna('').astype(str)
        d=d[d.Typrep.eq('A')]
    return d

def chunks(path, cols):
    if path.suffix=='.csv':
        yield from pd.read_csv(path,usecols=cols,dtype=str,chunksize=100000)
    else:
        import openpyxl
        w=openpyxl.load_workbook(path,read_only=True,data_only=True)
        s=w.worksheets[0];s.reset_dimensions();rows=s.iter_rows(values_only=True)
        header=next(rows);loc=[header.index(c) for c in cols];buffer=[]
        for r in rows:
            buffer.append([r[i] if i<len(r) else None for i in loc])
            if len(buffer)==20000:
                yield pd.DataFrame(buffer,columns=cols);buffer=[]
        if buffer:yield pd.DataFrame(buffer,columns=cols)
        w.close()

def main():
    a=argparse.ArgumentParser();a.add_argument('--source',required=True);a.add_argument('--kind',choices=['market','industry'],required=True)
    a=a.parse_args();path=Path(a.source);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    t=time.perf_counter();sig=sha(path);writer=None;n=0;low=None;high=None
    try:
        for d in chunks(path,MARKET if a.kind=='market' else INDUSTRY):
            d=normalize(d,a.kind)
            if not len(d):continue
            if a.kind=='market':
                low=min(low,d.Trddt.min()) if low is not None else d.Trddt.min()
                high=max(high,d.Trddt.max()) if high is not None else d.Trddt.max()
            table=pa.Table.from_pandas(d,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(out/'data_PRIVATE.parquet',table.schema,compression='zstd')
            writer.write_table(table);n+=len(d)
    finally:
        if writer:writer.close()
    receipt=dict(source=str(path),source_sha256=sig,kind=a.kind,rows=n,date_min=str(low),date_max=str(high),
                 seconds=time.perf_counter()-t,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 output_sha256=sha(out/'data_PRIVATE.parquet'))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
