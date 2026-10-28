"""Count synthetic Python forest fit calls in the frozen test suite, without changing its math."""
import json,os,time
from pathlib import Path
COUNTS=[]
def pytest_sessionstart(session):
 from crforest import CompetingRiskForest
 original=CompetingRiskForest.fit
 def traced(self,*args,**kwargs):
  row=dict(records=len(args[0]) if args else None,n_trees=self.config.n_trees,status='ENTERED');COUNTS.append(row)
  try:
   result=original(self,*args,**kwargs);row['status']='SUCCESS';return result
  except Exception:
   row['status']='RAISED';raise
 CompetingRiskForest.fit=traced

def pytest_sessionfinish(session,exitstatus):
 p=Path(os.environ['FIT_AUDIT_PATH']);p.write_text(json.dumps(dict(python_fit_calls=len(COUNTS),successful_python_fits=sum(x['status']=='SUCCESS' for x in COUNTS),rejected_validation_calls=sum(x['status']=='RAISED' for x in COUNTS),calls=COUNTS,pytest_exitstatus=exitstatus),indent=2))
