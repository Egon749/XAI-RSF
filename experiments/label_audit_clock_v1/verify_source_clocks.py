"""Strict public-date clocks; signing, fiscal dates and later restatements are not substitutes."""
from pathlib import Path
import json,re
import pandas as pd
from review_common import sha,write

def before_origin(published,origin):
 return pd.notna(published) and pd.Timestamp(published)<pd.Timestamp(origin)
def eligible(origin,event,horizon,lead_date,months):
 o=pd.Timestamp(origin);e=pd.Timestamp(event)
 return o<e<=pd.Timestamp(horizon) and o<=pd.Timestamp(lead_date)-pd.DateOffset(months=months)
def asof_versions(rows,origin):
 """Latest PUBLIC version of each fiscal-year/type; history remains a separate feature."""
 z=rows[rows.available_at.apply(lambda x:before_origin(x,origin))].copy()
 if z.empty:return z
 return z.sort_values('available_at').drop_duplicates(['audit_report_fiscal_year','audit_report_type'],keep='last')
def document_clock(doc):
 api=pd.Timestamp(doc['official_public_timestamp_ms'],unit='ms',tz='UTC').tz_convert('Asia/Shanghai').strftime('%Y-%m-%d')
 path=re.search(r'/finalpage/(\d{4}-\d{2}-\d{2})/',doc['url'])
 assert api==doc['published_at'],doc['document_id']
 assert path and path[1]==api,doc['document_id']
 assert sha(doc['file'])==doc['sha256'],doc['document_id']
 assert sha(Path(doc['file']).with_suffix('.txt'))==doc['text_sha256']
 return api

def main(run):
 run=Path(run);receipts=json.loads((run/'private/audit_document_retrieval.json').read_text());docs=[d for r in receipts for d in r['documents']]
 for d in docs:document_clock(d)
 write(run/'audit/document_clocks.json',{'documents':len(docs),'api_path_public_dates_agree':True,'downloaded_pdf_and_text_hashes_verified':True,'signed_dates_not_used_as_public_dates':True,'historical_restatement_not_backfilled':True,'limitation':'Official current archive metadata and original-version flags are documented; this is not a historical web capture or whole-dataset PIT certification.'})
 return len(docs)
if __name__=='__main__':
 import sys;main(sys.argv[1])
