"""Seal reporting artifacts after numerical and independent metric validation."""
import shutil,subprocess
from .run import RUN,PUBLIC
from experiments.post_st_dropcolumn_selection_v1.common import load,sha
import json


def main():
    for name in ['RESULTS_ZH.md','METHOD_INNOVATION_DECISION_ZH.md','PAPER_METHOD_RESULTS_ZH.md','DATA_AND_PROTOCOL_AUDIT_ZH.md','POINT_CANDIDATES.csv','GUARD_FAILURE_COUNTS.csv','GROUP_GUARD_SUMMARY.csv']:
        shutil.copy2(PUBLIC/name,RUN/'reports'/name)
    for p in (RUN/'audit').glob('pytest*.log'):shutil.copy2(p,PUBLIC/'audit'/p.name)
    receipt=load(PUBLIC/'execution_receipt.json')
    receipt.update(analysis_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        paired_coverage_validation=load(PUBLIC/'audit/paired_coverage_validation.json'),
        independent_metrics_validation=load(PUBLIC/'audit/final_validation.json'),
        metadata_correction='Frozen pairing.features_sha256 is the counterfactual reduced-column default pool. Actual explicit pools are retained in coverage.json and model tree_feature_sets. ACTUAL_SCHEDULE_HASHES.csv.gz independently replays actual pools; final code clarifies names. No fit/prediction/threshold change or rerun.',
        output_hashes={p.name:sha(p) for p in (RUN/'reports').iterdir() if p.is_file()},
        analysis_source_hashes={p.name:sha(p) for p in PUBLIC.glob('*.py')})
    for p in [PUBLIC/'execution_receipt.json',RUN/'audit/execution_receipt.json']:
        p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(dict(status=receipt['status'],sealed_outputs=len(receipt['output_hashes'])))

if __name__=='__main__':main()
