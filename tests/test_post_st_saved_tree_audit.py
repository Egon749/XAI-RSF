import numpy as np
from experiments.post_st_rule_features_v1.audit_mechanisms import tree_predictions


def test_saved_leaf_traversal_handles_missing_and_each_calendar_horizon():
    # Deliberately different endpoint grids and a horizon before any event.
    nodes=[dict(feature=0,threshold=1.,missing_left=True,left=1,right=2),
           dict(cif=[[.1,.2],[.2,.4]]),dict(cif=[[.3,.1],[.4,.2]])]
    result=tree_predictions(nodes,np.array([[np.nan],[2.],[0.],[2.]]),
                            [100.,200.],np.array([200.,100.,50.,200.]))
    np.testing.assert_allclose(result,[[.2,.4],[.3,.1],[0.,0.],[.4,.2]])
