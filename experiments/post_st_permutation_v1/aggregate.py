"""Verify frozen permutation artifacts and export company-free diagnostics."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata
from .permutation import B, SCHEMES, WEIGHTS, sha, write


def tail_summary(null, observed, tolerance=1e-12):
    """Plus-one Monte Carlo p; interval describes MC tail sampling, not prediction."""
    null = np.asarray(null)
    if null.ndim != 1 or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError('Expected finite scalar statistic and one-dimensional null')
    n = len(null)
    if not n:
        raise ValueError('Empty null distribution')
    exceed = int(np.count_nonzero(null >= observed - tolerance))
    lo = 0. if exceed == 0 else float(beta.ppf(.025, exceed, n - exceed + 1))
    hi = 1. if exceed == n else float(beta.ppf(.975, exceed + 1, n - exceed))
    return dict(exceedances=exceed, p=(exceed + 1) / (n + 1),
                null_q95=float(np.quantile(null, .95, method='higher')),
                mc_tail_probability_ci_low=lo, mc_tail_probability_ci_high=hi)


def validate_draws(a, p):
    if not np.array_equal(a['replicate'], np.arange(B)):
        raise ValueError('Missing, repeated or reordered replicate IDs')
    if a['feature_max'].shape != (B, p, 3) or a['global_max'].shape != (B, 3):
        raise ValueError('Unexpected score shape')
    if not np.isfinite(a['feature_max']).all() or (a['feature_max'] < 0).any():
        raise ValueError('Invalid scores')
    np.testing.assert_array_equal(a['feature_max'].max(axis=1), a['global_max'])


def aggregate(run, out):
    out.mkdir(parents=True, exist_ok=False)
    inp = run / 'input'
    protocol_path = inp / 'protocol.json'
    plan = json.loads(protocol_path.read_text())
    local_plan = Path(__file__).with_name('frozen_protocol.json')
    assert sha(protocol_path) == sha(local_plan)
    assert plan['permutations_per_scheme'] == B and plan['weight_order'] == list(WEIGHTS)
    assert plan['schemes'] == list(SCHEMES)
    assert sha(inp / 'permutation_input_PRIVATE.npz') == plan['input_sha256']
    assert sha(inp / 'code/permutation.py') == plan['script_sha256']
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == expected
    with np.load(inp / 'permutation_input_PRIVATE.npz', allow_pickle=False) as z:
        names = z['feature_names'].copy()
        observed = z['observed_feature_max'].copy()
        has_candidate = np.isin(np.arange(len(names)), z['features'])
        cohorts = []
        for year in np.unique(z['year']):
            events = z['event'][z['year'] == year]
            cohorts.append(dict(entry_year=int(year), companies=len(events),
                                termination=int((events == 1).sum()), recovery=int((events == 2).sum()),
                                event_zero=int((events == 0).sum())))
    pd.DataFrame(cohorts).to_csv(out / 'training_cohort_counts.csv', index=False)
    price = int(np.flatnonzero(names == 'log_price_last')[0])
    rows, draw_rows, receipts, global_rows = [], [], [], []
    for scheme in SCHEMES:
        directory = run / 'results/local' / scheme
        receipt = json.loads((directory / 'receipt.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['scheme'] == scheme
        assert (receipt['start'], receipt['stop'], receipt['draws']) == (0, B, B)
        assert receipt['new_research_fits'] == 0
        assert receipt['protocol_sha256'] == sha(protocol_path)
        assert receipt['script_sha256'] == plan['script_sha256']
        path = directory / 'null_scores_PRIVATE.npz'
        assert sha(path) == receipt['output_sha256']
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}
        validate_draws(a, len(names))
        assert a['valid_candidates'].shape == (B,)
        assert ((a['valid_candidates'] >= 0) &
                (a['valid_candidates'] <= plan['observed_spatial_candidates'])).all()
        joint_max = a['global_max'].max(axis=1)
        receipts.append(receipt)
        for k, weight in enumerate(WEIGHTS):
            ranks = rankdata(-observed[has_candidate, k], method='min')
            rank_map = dict(zip(np.flatnonzero(has_candidate), ranks.astype(int)))
            for j, name in enumerate(names):
                row = dict(scheme=scheme, weight=weight, feature=j, name=name,
                           has_observed_candidate=bool(has_candidate[j]),
                           observed_score=float(observed[j, k]), observed_rank=rank_map.get(j))
                if has_candidate[j]:
                    for family, null in [('own_feature', a['feature_max'][:, j, k]),
                                         ('all_variables', a['global_max'][:, k]),
                                         ('variables_and_weights', joint_max)]:
                        row.update({f'{family}_{key}': value for key, value in
                                    tail_summary(null, observed[j, k]).items()})
                rows.append(row)
            winner = int(np.argmax(observed[:, k]))
            global_rows.append(dict(scheme=scheme, weight=weight, winner=names[winner],
                                    observed_max=float(observed[winner, k]),
                                    **tail_summary(a['global_max'][:, k], observed[winner, k])))
        for b in range(B):
            row = dict(scheme=scheme, replicate=b, admissible_candidates=int(a['valid_candidates'][b]),
                       joint_max=float(joint_max[b]))
            for k, weight in enumerate(WEIGHTS):
                row[f'max_{weight}'] = float(a['global_max'][b, k])
                row[f'price_{weight}'] = float(a['feature_max'][b, price, k])
            draw_rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'feature_tests.csv', index=False)
    draws = pd.DataFrame(draw_rows)
    draws.to_csv(out / 'null_maxima.csv', index=False)
    pd.DataFrame(global_rows).to_csv(out / 'global_tests.csv', index=False)
    selected = table[(table.name == 'log_price_last') |
                     ((table.weight == 'combined') & (table.observed_rank <= 5)) |
                     ((table.weight != 'combined') & (table.observed_rank == 1))]
    selected.to_csv(out / 'selected_comparisons.csv', index=False)
    summary = dict(status='COMPLETE', schemes=len(SCHEMES), draws_total=len(draws),
                   new_research_fits=0, companies=plan['training_companies'],
                   protocol_sha256=sha(protocol_path), workers=receipts,
                   total_worker_seconds=sum(r['seconds'] for r in receipts),
                   outcome='Exploratory global-null diagnostic, not a noise-dominance or generalization test',
                   result_hashes={p.name: sha(p) for p in sorted(out.glob('*.csv'))})
    write(out / 'summary.json', summary)
    plot(draws, table, out)
    print(selected[['scheme', 'weight', 'name', 'observed_score', 'own_feature_p',
                    'all_variables_null_q95', 'all_variables_p', 'variables_and_weights_p']].to_string(index=False))


def plot(draws, table, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False,
                         'font.size': 11, 'svg.hashsalt': 'post_st_permutation_v1'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex='col')
    for row, scheme in enumerate(SCHEMES):
        d = draws[draws.scheme == scheme]
        for col, weight in enumerate(WEIGHTS[:2]):
            ax = axes[row, col]
            null = d[f'max_{weight}'].to_numpy()
            subset = table[(table.scheme == scheme) & (table.weight == weight)]
            price = subset[subset.name == 'log_price_last'].iloc[0]
            win = subset[subset.observed_rank == 1].iloc[0]
            q95 = np.quantile(null, .95, method='higher')
            ax.hist(null, bins=24, color='#c5d7e7', edgecolor='white', label='全变量最大分数零分布')
            ax.axvline(q95, color='#515966', linestyle='--', label=f'零分布95%分位 {q95:.2f}')
            ax.axvline(price.observed_score, color='#df8540', linewidth=2,
                       label=f'股价 {price.observed_score:.2f}；p={price.all_variables_p:.3f}')
            ax.axvline(win.observed_score, color='#246899', linewidth=2,
                       label=f'实际最高 {win.observed_score:.2f}；p={win.all_variables_p:.3f}')
            mode = '全体成对置换' if row == 0 else '入组年内成对置换'
            weight_name = '默认 (1,1)' if col == 0 else '只看退市 (1,0)'
            ax.set_title(f'{mode} · {weight_name}')
            ax.set_ylabel('置换次数')
            ax.set_xlabel('根节点分裂分数')
            ax.legend(fontsize=8.7, loc='upper right', framealpha=.88)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('实际根节点分数与置换最大分数的比较', fontsize=15)
    fig.text(.5, .015, '每种方案500次；309家公司。p按全变量最大值校准。默认最高为财报陈旧度，退市最高为现金比率。\n'
             '固定设计、成对置换随访时间与事件；非拒绝不等于证明“全是噪声”，拒绝不等于预测有效。',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(out / 'root_maxima.png', dpi=170)
    fig.savefig(out / 'root_maxima.svg', metadata={'Date': None})
    plt.close(fig)
    p = out / 'root_maxima.svg'
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.run, args.out)
