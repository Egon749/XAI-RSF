"""Freeze provenance, historical admissibility, and finite search budget."""
import argparse
from dataclasses import asdict
import datetime
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from experiments.post_st_dropcolumn_selection_v1.common import load, sha, write, arrays, configuration
from .search import FULL, LEGACY_REMOVED, historical_removed, lattice

NAME = 'post_st_feature_combination_search_v1'


def prepare(repo):
    public = repo / 'experiments' / NAME
    location = load(public / 'run_location.json')
    run, parent, previous = map(Path, [location['private_run'], location['parent_private_run'], location['previous_h24_run']])
    assert not (public / 'frozen_protocol.json').exists()
    assert sha(repo / 'CURRENT_POST_ST_MODEL.json') == 'aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65'
    classification = pd.read_csv(repo / 'experiments/post_st_feature_permutation_v1/results/feature_classification.csv')
    features = load(parent / 'input/features.json')
    assert len(features) == 83
    support = pd.read_csv(parent / 'reports/historical_support.csv')
    sources = {}
    for path in sorted((parent / 'input').rglob('*.npz')):
        # Read-only provenance hashing does not feed any outcome to selection.
        sources[str(path)] = sha(path)
    for path in [parent / 'input/group_specs.json', parent / 'input/features.json',
                 parent / 'input/subsets.json', parent / 'input/arms.json',
                 parent / 'reports/historical_support.csv',
                 repo / 'experiments/post_st_feature_permutation_v1/results/feature_classification.csv']:
        sources[str(path)] = sha(path)
    # Verify the diagnosis at the historical information date, not its later publication date.
    diag = []
    for year in [2023, 2024, 2025]:
        historical = classification[classification.outer_cutoff == min(year, 2024)]
        actual = set(historical.loc[historical.subtype == 'suspected_redundant', 'feature'].astype(int))
        assert actual == set(historical_removed(year)), (year, actual)
        for j in LEGACY_REMOVED:
            diag.append(dict(year=year, feature_id=j, feature_name=features[j]['feature_name'],
                             label=features[j]['label'], diagnosis_available=j in actual,
                             original_diagnosis_cutoff=min(year, 2024)))
    protocol = dict(
        name=NAME, status='FROZEN_BEFORE_NEW_FITS_AND_SEARCH_SCORES',
        source_commit='722089d7fccf362df478997d1aba08b6b326844c',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **location, years=[2023, 2024, 2025], reference=asdict(configuration(42, 83)),
        seeds=[42, 101, 202], trees_per_seed=32, parent_AJ_strength=10,
        data='Unchanged old conservative C83; next trading-day pre-open; announcement-date first final termination cause1 versus first full recovery cause2; train follow-up cap24 calendar months',
        objective='Historical equal-cohort mean AP24, then annual-quota capture24, then smaller complete function spatial cells; same H24 safeguards as previous experiment against full83, not against the last step',
        support_gate=load(previous / 'input/protocol.json')['support_gate'],
        constraints=load(previous / 'input/protocol.json')['constraints'],
        historical74='Complete fixed74 is a mandatory anchor in2024/2025. In2023 only8 of the9 diagnoses existed: mandatory anchors83/75. Historical fixed74 remains a retrospective external comparator, not an injected2023 seed. A general neighbourhood may independently generate the same subset by symmetric enumeration of ALL variables; record that provenance, without claiming the later diagnosis was available.',
        initial_search='Enumerate all2^8=256 /2^9=512 /2^9=512 combinations of historically supported redundant fields. Add every legal existing single-drop/group candidate and both anchors. Preserve audit76..82 as an atomic block. No future outcome or future diagnosis selects the2023 lattice.',
        expansion=dict(rounds=2, centers=4,
                       center_rule='Mandatory full83 and historical74/75 anchor; best raw historical AP24 and best feasible candidate, deterministic fill if duplicates',
                       moves='Every toggle of original inputs0..75, atomic toggle of audit76..82, and each frozen historical correlation group replaced by all/none/one/two/all single representatives',
                       traversal='Do not require an intermediate move to satisfy final guards; permits non-monotone paths. Score all distinct candidates, never stop early because the current path has no acceptable deletion.'),
        final_decision='Highest-ranked evaluated candidate passing all unchanged safeguards against83, only if historical support passes; otherwise retain83. A candidate being mandatory does not exempt it from safeguards. Record scores/rejections for83 and74/75 explicitly.',
        search_claim='Complete enumeration only within the8/9-dimensional redundancy lattice; two finite neighbourhood rounds elsewhere. No global optimum over2^83 subsets is claimed.',
        uncertainty=dict(paired_outer_repeats=1000, seed=2026110174,
                         scope='Company resampling stratified by evaluation year, common indices, fixed predictions/IPCW. Descriptive intervals do not include training/search uncertainty or guarantee multiplicity control.'),
        stability=dict(final_fixed_set_company_refits=50,
                       description='Use the same50 historical company draws as prior experiment. Refit selected fixed feature set, compare to reused full83 bootstrap models. This checks model/function stability CONDITIONAL ON the selected set; it does not rerun the expanded search or establish selection stability.'),
        budget='Initial lattice plus all legal existing diagnostics; exactlytwo expansion rounds, four centers each. Independent bundles of3forests, at most4bundles per Slurm task. No additional search after outer results. At most450 final fixed-set bootstrap forest fits. Four pilot bundles (12forests) are retained and reused when identical.',
        numerical='Every new raw/shrunk forest and three-seed merged predictor: q<=2, nonnegative/monotone/sum<=1 CIF, save/reload, explicit complete functions on ALL training/eval rows/native times/both causes, error<=1e-10; no clipping',
        outer='Only after decisions sealed, evaluate2023/2024/2025 at12/24months, both causes, annual/pooled;2025H24 unsupported NA. Include original83, legacy74, previous automatic and current selected. All cohorts already exposed, exploratory; no reference replacement.',
        inherited_limits='Original annual-value vintage PIT remains PROVISIONAL; no new availability or version assumption. Retain all failures, candidates, seeds and private parameters.',
        resources='Two-node existing Slurm/NFS. Single-thread workers,4GiB address-space limit; record real peakRSS/affinity/times. Do not alter cgroups, global pool config, tree settings, candidate proportions or data.'
    )
    write(public / 'frozen_protocol.json', protocol)
    write(run / 'input/protocol.json', protocol)
    write(run / 'audit/source_hashes.json', sources)
    write(run / 'audit/preparation.json', dict(status='AUDITED_PROVISIONAL', reference_unchanged=True,
                                               input_files=len(sources), source_commit=protocol['source_commit'],
                                               directory_incident='Initial local mkdtemp waited on existing NFS directory lease; terminated only that waiting preparer and created unique empty directory from lemon. No fit had started; no service/config change.'))
    pd.DataFrame(diag).to_csv(public / 'HISTORICAL_CANDIDATE_AVAILABILITY.csv', index=False)
    support.to_csv(public / 'HISTORICAL_EVENT_SUPPORT.csv', index=False)
    shutil.copy2(parent / 'input/features.json', public / 'features.json')
    audit = '''# 数据、候选与时间边界审计

本轮承接722089d，不修改当前研究参考、旧输入、旧模型或历史结果。83列名称、顺序、缺失语义与原保守C完全沿用；森林q2、深度4、每种子32树、种子42/101/202、每树25列、共享切点与直接raw父节点AJ收缩10保持。

入组为首次财务/审计ST公告后下一交易日开盘前。终点为公告日先最终退市（原因1）或先完整恢复（原因2）。各年训练/标签严格截止当年1月1日之前；24日历月训练上限不变。企业训练数309/345/390，已知退市14/26/49，恢复184/198/211；外层36/45/53家公司，训练评价公司无重叠。

原始披露版本仍为PROVISIONAL；本轮未重新认证25份年报，也没有采用后来接受版本的输入。全部私有输入做SHA256冻结。准备期间共享根目录的NFS租约造成一次等待，已仅停止本次等待的准备进程，并通过现有lemon共享挂载创建新目录；未重启NFS或更改旧数据。

## 74项的历史可用性

2023截止原诊断支持8项，2024支持9项；差异为58号近3个月总市值变化。因此2023的显式起点为83和75，2024/2025为83和74。2023后验74固定结果保留对照，不当作当时已知先验。所有列的通用邻域枚举可能独立生成同一个74项子集，来源需如实记录；不能把这种搜索结果说成原9项诊断在2023已经齐全。

两种起点都保留到每轮扩展，并且所有最终候选与原83比较。74可以不满足保护条件，但不能不经比较就遗漏。支持门槛保持：H24至少两个成熟有效队列、至少10例目标事件；各队列至少3例，历史拟合至少1例退市。2023仍只有1队列6例，搜索可以提供诊断，但最终自动决策必须回退83。2024/2025为13/22例，达到操作门槛不代表功效充分。

## 搜索边界

先完整枚举8/9项冗余字段的保留删除组合，再加入原全单删/相关组候选；随后两轮、每轮四个起点，枚举全部原变量的增删和历史相关组代表替换。中间步骤允许暂时不合格，最终接受仍需满足冻结的预测及解释保护。不改变门槛迎合结果，不将有限搜索称作全局最优。

2023–2025外层已暴露，所有外层结果都仅作探索性复核。选择器只接收截止点前的历史预测、标签和组定义；2025外层H24截至2026-06-30随访不足，保持NA。最终固定特征集50次公司重训用于函数/模型稳定性，不冒充完整搜索的重抽样稳定性。
'''
    (public / 'DATA_AND_PROTOCOL_AUDIT_ZH.md').write_text(audit)
    print(json.dumps(dict(run=str(run), candidate_lattice_sizes={y:len(lattice(y)) for y in [2023,2024,2025]}, hashed_inputs=len(sources))))


if __name__ == '__main__':
    prepare(Path(__file__).resolve().parents[2])
