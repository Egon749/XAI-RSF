"""Replay finite analysis from preserved private evidence. Never retrieves or fits by default."""
from pathlib import Path
import argparse,json,time
from review_common import sha,write

def verify_originals(run):
 run=Path(run);before=json.loads((run/'audit/sources_before.json').read_text());after={};changed=[]
 for p,v in before.items():
  after[p]={'sha256':sha(p),'bytes':Path(p).stat().st_size}
  if after[p]!=v:changed.append(p)
 write(run/'audit/sources_after.json',after)
 receipt={'files':len(before),'changed_files':changed,'all_original_sources_unchanged':not changed,'sha256':True}
 write(run/'audit/immutability_verification.json',receipt)
 assert not changed,'Original inputs changed; stop review'
 return receipt

def main(run):
 run=Path(run);start=time.monotonic();records=[]
 for name in ['adjudicate_labels','validate_label_patch','audit_reason_mapping','compare_frozen_captures','verify_source_clocks','build_audit_timeline','join_prediction_origins','check_input_representation','summarize_evidence']:
  module=__import__(name);t=time.monotonic()
  try:module.main(run);records.append(dict(stage=name,status='PASS',seconds=time.monotonic()-t))
  except Exception as ex:
   records.append(dict(stage=name,status='FAIL',error=repr(ex)));write(run/'audit/analysis_stages.json',records);raise
 write(run/'audit/analysis_stages.json',records);verify_originals(run)
 write(run/'audit/replay_receipt.json',{'stages':len(records),'elapsed_seconds':time.monotonic()-start,'prediction_model_fits':0,'calibration_fits':0,'threshold_choices':0,'hyperparameter_searches':0,'private_annotations_required':True,'network_downloads_during_replay':0})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main(a.run)
