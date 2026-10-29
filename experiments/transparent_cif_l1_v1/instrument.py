"""Observational wrappers only: count calls/costs without changing values or RNG."""
import time,json,os
from pathlib import Path
CALLS=[]
def wrap(owner,name,kind):
 old=getattr(owner,name)
 def traced(*a,**kw):
  row={'kind':kind,'status':'STARTED'};CALLS.append(row);t=time.perf_counter()
  if kind=='forest_fit':row.update(records=len(a[1]),trees=a[0].config.n_trees)
  if kind=='ebm_fit':row.update(records=len(a[1]),rounds=a[0].max_rounds,interactions=a[0].interactions)
  try:
   result=old(*a,**kw);row['status']='SUCCESS'
   if kind=='pair_search':row['added_trees']=result[1]['added_trees']
   if kind=='ordinary_extension':row['added_trees']=len(result.trees_)-len(a[0].trees_)
   return result
  except Exception as e:row.update(status='RAISED',exception=type(e).__name__);raise
  finally:row['seconds']=time.perf_counter()-t
 setattr(owner,name,traced)
def install():
 from crforest import CompetingRiskForest
 import crforest.pair_candidates as pair,crforest.transparent as tr
 from interpret.glassbox import ExplainableBoostingClassifier
 wrap(CompetingRiskForest,'fit','forest_fit');wrap(pair,'augment_pair_bank','pair_search');wrap(pair,'extend_ordinary_bank','ordinary_extension');wrap(tr,'solve_simplex_brier','weight_solve');wrap(ExplainableBoostingClassifier,'fit','ebm_fit')
def save(path):Path(path).write_text(json.dumps(CALLS,indent=2))
def pytest_sessionstart(session):install()
def pytest_sessionfinish(session,exitstatus):save(os.environ['CALL_AUDIT'])
