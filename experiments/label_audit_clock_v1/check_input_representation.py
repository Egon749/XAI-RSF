"""Inspect frozen feature names and their source constructions; no opinion imputation."""
from pathlib import Path
import json,numpy as np
from review_common import BASE,register,write

def main(run):
 run=Path(run);fields=BASE/'input/code/experiments/shared_partition_v1/input_fields.json'
 prepare=Path('/home/oliver/LLMworkspace/early-warning-12m-server-isolated/experiments/early_warning_12m/prepare.py')
 formula=prepare.parent/'source_reference/financial_formulas.py'
 metadata=Path('/home/oliver/pylib/ChinaData_WRDS/metadata')
 register(run,[fields,prepare,formula,metadata/'research_domain_availability_audit.csv',metadata/'regulatory_event_field_audit.csv'])
 f=json.loads(fields.read_text());assert f['count']==75 and len(f['ordered_names'])==75
 z=np.load(BASE/'input/final_train_PRIVATE.npz');assert z['X'].shape[1]==75
 body=formula.read_text();assert 'out["loss_flag"] = net_profit.lt(0)' in body and 'out["negative_equity_flag"] = equity.lt(0)' in body
 assert not any(any(w in name.lower() for w in ['audit','opinion','remediation']) for name in f['ordered_names'])
 write(run/'audit/input_representation.json',{'model_inputs':75,'content':72,'financial':48,'market':24,'quality':3,'names_and_source_constructions_reviewed':True,'explicit_financial_audit_opinion':False,'explicit_external_internal_control_opinion':False,'explicit_self_assessment':False,'explicit_remediation':False,'proxy_fields':['loss_flag','negative_equity_flag','interest_coverage','cfo_to_net_income','financial_disclosure_age'],'proxy_limitation':'Numeric distress and age proxies do not encode the auditor statement or its public timestamp; no row-level statistical equivalence is asserted.','M2_representation':'Original M2 uses 72 content +72 missing indicators +3 quality; full-sector reference, not matched-role manufacturing refit.','registry_feasibility':'BDT event-conditioned audit fields cannot replace a full company-year audit panel; annual opinions need public date/version and coverage independent of event status.'})
if __name__=='__main__':
 import sys;main(sys.argv[1])
