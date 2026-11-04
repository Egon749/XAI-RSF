import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .worker import response, rankings, run_one


def test_permutation_keeps_unknown_and_event_count():
    labels = np.r_[np.repeat(-1, 5), np.repeat(0, 32), np.repeat(1, 5)]
    for rep in range(1, 201):
        y = response(labels, 2023, rep)
        np.testing.assert_array_equal(labels == -1, y == -1)
        assert sum(y == 1) == 5
        np.testing.assert_array_equal(y, response(labels, 2023, rep))
    np.testing.assert_array_equal(labels, response(labels, 2023, 0))


def test_weighted_scores_with_ties():
    y = np.array([0,0,1,1,0,1])
    p = np.array([.2,.4,.2,.8,.3,.4])
    w = np.array([4.,1.,1.,.3,2.,3.])
    np.testing.assert_allclose(rankings(y,p,w), [average_precision_score(y,p,sample_weight=w),
                                                 roc_auc_score(y,p,sample_weight=w)])


def test_three_fold_scoring_and_true_prefixes(tmp_path):
    rng = np.random.default_rng(3)
    n = 48
    ev = np.tile([1,0,2,0], n//4)
    t = np.tile([6,8,5,12], n//4)
    d = dict(groups=np.arange(n), X=rng.normal(size=(n,74)), event=ev, analysis_time=t,
             label_eligible=np.ones(n,bool))
    res = run_one(d, 2023, 0, tmp_path)
    import pandas as pd
    tab = pd.read_csv(tmp_path/'rounds.csv')
    assert len(tab) == 48 and res['known'] == 36 and res['positives'] == 12
    np.testing.assert_allclose(tab[['fold1_ap','fold2_ap','fold3_ap']].mean(axis=1), tab.cv_ap)
