"""Publish anonymized aggregates only after tests, stage checks and immutable-source audit."""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
from review_common import sha,write
from run_review import verify_originals

def junit(path):
 root=ET.parse(path).getroot();suites=[root] if root.tag=='testsuite' else list(root)
 return {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
def main(run):
 run=Path(run);public=Path(__file__).parent;proof=verify_originals(run)
 new=junit(run/'audit/tests.xml');old=junit(run/'audit/old_no_fit_tests.xml');assert new['tests']==40 and old['tests']==23
 assert not any(x[k] for x in [new,old] for k in ['failures','errors','skipped'])
 stages=json.loads((run/'audit/analysis_stages.json').read_text());assert len(stages)==9 and all(x['status']=='PASS' for x in stages)
 out=public/'results';out.mkdir(exist_ok=True)
 for f in (run/'results').glob('*.csv'):shutil.copyfile(f,out/f.name)
 evidence=json.loads((run/'audit/evidence_counts.json').read_text())
 private_outputs={str(p.relative_to(run)):sha(p) for p in (run/'private').glob('*') if p.is_file() and (p.suffix in ['.parquet','.json'])}
 write(run/'audit/private_output_hashes.json',private_outputs)
 # Aggregate-only receipt: no identities, excerpts, thresholds or model parameters.
 receipt=dict(status='COMPLETED_NO_FIT_WITH_EXPLICIT_EVIDENCE_GAPS',data_status=['PROVISIONAL','POST_EXPOSURE_EXPLORATORY'],branch='research/label-audit-clock-v1',policy_reference_sha='847619a177cd3ee51215c846009919d540d46cef',manufacturing_reference_sha='5de9cf55a40de6b8af41a6877e51ab3529c6c003',code_revision='The containing Git commit identifies this receipt and code. Final full SHA and remote equality are separately saved in private audit/git_publication.json.',private_run=str(run),completed_analysis_stages=9,new_prediction_model_fits=0,new_calibration_fits=0,new_threshold_choices=0,new_hyperparameter_searches=0,tests=dict(new=new,old_no_fit=old,old_deselected=1,old_deselected_reason='Would write to prior run; replaced by new-directory immutable audit.',historical_regression_failures=1,regression_fixed='Calendar month-end direction now exactly follows original event_leads; actual counts unchanged.',old_full_fit_suite_run=False),label_patch=json.loads((run/'audit/label_patch_validation.json').read_text()),reason_mapping=json.loads((run/'audit/reason_mapping_result.json').read_text()),cohort_evidence=evidence,capture_reproduction=json.loads((run/'audit/capture_reproduction.json').read_text()),origin_join=json.loads((run/'audit/origin_join.json').read_text()),source_integrity=proof,source_manifest_before_sha256=sha(run/'audit/sources_before.json'),source_manifest_after_sha256=sha(run/'audit/sources_after.json'),new_evidence_manifest_sha256=sha(run/'audit/new_evidence_manifest.json'),private_output_manifest_sha256=sha(run/'audit/private_output_hashes.json'),label_patch_sha256=sha(run/'private/label_patch_PRIVATE.parquet'),review_annotations_sha256=sha(run/'private/audit_review_annotations.json'),limitations=['47 same-branch events remain unresolved; not 47 confirmed errors.','Prior financial/IC opinion and other subtype coverage gaps retained.','No full-riskset audit panel with verified public clocks; event-conditioned vendor fields cannot substitute.','Current archive dates do not constitute full historical-vintage PIT certification.','No performance under revised labels or new audit inputs estimated.'],decisions=['LABEL_PATCH_READY_NOT_APPLIED','AUDIT_DISCLOSURE_TOO_LATE_FOR_DECLARED_LEAD','EXISTING_INPUT_MODEL_GAP','DATA_EVIDENCE_INCOMPLETE'],next_stage_executed=False)
 write(public/'execution_receipt.json',receipt);write(run/'audit/execution_receipt.json',receipt)
 print(json.dumps({'stages':9,'tests_passed':new['tests']+old['tests'],'original_files_unchanged':proof['files'],'fits':0}))
if __name__=='__main__':
 import sys;main(sys.argv[1])
