"""No predictive features generated here: audited building blocks for a later protocol."""
import math
from asof_policy_resolver import fiscal_applicability
UNITS={'yuan':1.,'CNY':1.,'万元':1e4,'亿元':1e8}
def yuan(value,unit):
 if unit not in UNITS:raise ValueError('Unknown currency/unit')
 return value*UNITS[unit]
def evaluate(expression,values):
 """Three-valued logic; never use eval; missing is not False."""
 if expression is None:return None
 if 'and' in expression:
  v=[evaluate(x,values) for x in expression['and']];return False if False in v else None if None in v else True
 if 'or' in expression:
  v=[evaluate(x,values) for x in expression['or']];return True if True in v else None if None in v else False
 x=values.get(expression['field']);y=expression['value'];op=expression['op']
 if x is None or isinstance(x,float) and math.isnan(x):return None
 if op=='in':return x in y
 return {'<':lambda:x<y,'<=':lambda:x<=y,'>':lambda:x>y,'>=':lambda:x>=y,'==':lambda:x==y}[op]()
def dependencies(expression):
 if expression is None:return set()
 if 'field' in expression:return {expression['field']}
 return set().union(*(dependencies(x) for x in expression.get('and',expression.get('or',[]))))
def feature_status(rule,values,fiscal_year,report_type,transition_resolved=True):
 a=fiscal_applicability(rule,fiscal_year,report_type,transition_resolved)
 if a!='POTENTIALLY_APPLICABLE':return a,None
 expr=rule['expression'];v=evaluate(expr,values)
 return ('MISSING' if v is None else 'OBSERVED'),v
