"""Publish aggregate evidence only; company rows and parameters remain private."""
from pathlib import Path
import argparse,datetime,importlib.metadata,json,platform,shutil,sys
import numpy as np
import pandas as pd
from .common import *


def table(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)):return '—' if not np.isfinite(v) else f'{v:.5f}'
        return str(v).replace('|','／').replace('\n',' ')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def report(run,repo):
    public=repo/'experiments'/NAME;out=public/'results';out.mkdir(exist_ok=True)
    for p in (run/'reports').glob('*.csv'):shutil.copy2(p,out/p.name)
    for n in ['freeze.json','preparation.json','retry_equivalence.json','analysis_validation.json','independent_validation.json','current_reference_unchanged.json','failure_summary.json','formal_gate.json']:
        shutil.copy2(run/'audit'/n,out/n)
    logs=public/'logs';logs.mkdir(exist_ok=True)
    for n in ['full_tests_pre_run.log','full_tests_with_R.log','full_tests_pass.log','full_tests_frozen.log','full_tests_final.log','analysis.log','independent_validation.log']:
        if (run/'audit'/n).exists():shutil.copy2(run/'audit'/n,logs/n)
    d=pd.read_csv(out/'feature_classification.csv');a=pd.read_csv(out/'permutation_temporal_summary.csv',dtype={'key':str})
    base=pd.read_csv(out/'baseline_metrics.csv');support=pd.read_csv(out/'cohort_support.csv');val=pd.read_csv(out/'model_validation.csv')
    current=d[d.outer_cutoff==2024];retain=current[current.category=='保留'];redundant=current[current.category=='疑似冗余或噪声']
    counts=d.groupby(['outer_cutoff','category']).size().unstack(fill_value=0).reset_index()
    target=a[(a.outer_cutoff==2024)&(a.horizon==24)&(a.cause==1)]
    audit=target[(target.kind=='group')&(target.key=='76,77,78,79,80,81,82')].iloc[0]
    top=target[(target.kind=='single')&~target.linked_group].sort_values('delta_auc',ascending=False).head(10)
    pair=target[target.kind=='pair'];pairpass=int(pair.component_retention_evidence.sum())
    group_baseline=base[(base.outer_cutoff==2024)&(base.horizon==24)&(base.cause==1)&base.informative]
    gate=json.loads((run/'audit/formal_gate.json').read_text());check=json.loads((run/'audit/analysis_validation.json').read_text());ind=json.loads((run/'audit/independent_validation.json').read_text())
    freeze=json.loads((run/'audit/freeze.json').read_text());tasks=pd.read_csv(out/'execution_tasks.csv')
    text=f'''# 历史特征置换诊断结果

本轮没有重训或删变量。复用15个历史森林，完成40/40修复后批次、{check['unique_items']}个“历史队列×置换对象”、每对象128次置换。当前参考模型保持不变。

截至2024年初，83个变量中：**保留{len(retain)}项、疑似冗余或噪声{len(redundant)}项、证据不足{int((current.category=='证据不足').sum())}项**。保留项全部属于同一个7字段审计信息块，不能当成7个独立发现。疑似项本轮全部是相关性冗余候选，**没有变量满足疑似纯噪声门槛，也没有实际删除变量**。

## 时间范围与2023训练信息

当前2024参考森林已用345家公司、26例截至2024年初已知退市训练，包含2023入组企业。本轮不重复拟合，也不把该森林的2023在样本预测当验证。2023队列用2023年前拟合的已有森林预测，只读取2024-01-01之前的训练标签：36家公司、4例早期退市、最长8.677个月，12/24个月均无随访到期者，不能据此评价这两个完整终点。

用于截至2024年初24个月退市主要置换判断的有效窗口只有2020和2021：共99家、13例退市，对应模型拟合时仅有2例和3例退市。2019模型训练时0例退市；2022/2023队列在截止点没有24个月随访支持。**新增2023信息没有凭空增加完整的24个月样本外证据。**

{table(group_baseline[['cohort','n','train_events','validation_events','auc','ap','brier']])}

2023、2024截止结果分别如下，重叠企业不能视为两个独立研究：

{table(counts)}

## 审计信息块有保留依据，未识别单字段独立效果

七个字段共同交换企业归属后，2020/2021队列等权平均AUC下降 **{audit.delta_auc:.5f}**，描述性95%区间 **[{audit.auc_lo:.5f}, {audit.auc_hi:.5f}]**；AP平均下降{audit.delta_ap:.5f}，Brier平均增加{audit.delta_brier:.6f}。正的delta_AUC/AP表示置换损害排序，正的delta_Brier表示置换损害概率误差。

这些是同一组字段的共同结果。审计意见存在类别约束，分别打乱可能造出相互矛盾的意见标记，因此凡涉及该块的置换均共同移动全部7项。逐变量表保留`linked_group=true`，不能对这7项排序或把效果相加。

保留名单：{'、'.join(retain.label)}。

其他单变量中观测平均下降较大的项目如下。区间包括0的项目仍为“证据不足”；下表不是保留名单，不能只挑正值宣称有效。完整正负结果均在CSV中。

{table(top[['label','delta_auc','auc_lo','auc_hi','delta_ap','delta_brier']])}

最近股价等变量在旧模型中的置换效果弱，不能反推股价没有真实预测价值：这首先反映这些历史森林实际怎样使用变量，且其退市训练支持很弱。

## 九项疑似冗余，不能直接删除

{table(redundant[['label','reason']])}

相关性由每个滚动训练集单独计算，至少30个成对非缺失样本，|Spearman|≥0.95且至少三个训练窗口重复出现。选择观测覆盖较高者为对照代表，覆盖相同才按原字段序号打破平局。**这只是信息可能重复，不是噪声证明；相关组与单变量置换的结果均保留，尚未做删列重训。**

## 二元交互

已检查所有已存森林中的原生二元项、其训练和验证贡献幅度、空间单元公司/退市支持，以及成对联合置换。共同置换一对变量保留二者内部关系，其性能变化同时包含主效应及关联的其他交互，不能解释为纯二元交互增量。

另计算从完整预测中减去某个原生二元项的AUC/AP变化，仅称为“分数诊断”；该截断分数可能不再是合法CIF，不用它报告Brier或概率合法性，也不冒充重训消融。本轮24个月退市有 **{pairpass}个**二元项达到预设跨年份保留门槛。单年度正向变化不能取代重复证据。

## 数值、失败与计算

修复后Slurm作业{gate['job']}：40/40成功，oliver {gate['by_host'].get('oliverdesktop',0)}项、lemon {gate['by_host'].get('lemon',0)}项，计算{gate['seconds']:.2f}秒；单任务峰值RSS最大{tasks.peak_rss_MiB.max():.1f} MiB。调度器采用未校准启发式，不宣称这是最优资源布局。

全部训练/历史评价样本、原生时间及两个原因的完整函数和直接预测最大重构误差{val.full_reconstruction_error.max():.3e}，显式函数求和误差{val.component_sum_error.max():.3e}；全部置换的代数加速结果与直接森林最大误差{check['maximum_permutation_error']:.3e}。正式评分使用直接森林结果，避免不同求和顺序把相等风险分裂成微小差异。另用三个种子显式平均和sklearn独立复算{ind['independent_effect_rows']}行、每行128次置换，指标误差最大{ind['maximum_metric_difference']:.3e}。

完整测试187项通过，包含独立R检验。初次完整测试因未配置R而跳过5项；补上R可执行文件后仍因缺少既有库路径失败5项，正确设置EW_RLIB后全部通过。首轮Slurm2582的40个批次因二元变量索引被序列化为字符串而失败；代码修复后使用新独立目录重跑，协议、数据和随机抽样逐项核对不变，旧快照/失败记录保留。不是挑选成功种子，也没有失败森林拟合。

## 可以与不可以得出的结论

支持暂时保留审计信息块，并将9项高度相关指标列入未来去冗余候选。其余变量缺乏足够证据区分“没被这个模型学到”和“确实无用”；67项证据不足应暂时保留，不能解释为应删除67项。未发现可可靠归类为纯噪声的字段。

本轮没有证明筛选会提高当前2023/2024测试AUC，也没有生成新筛选模型。保留q2解释是数学表示性质，不等于变量及交互的真实作用已被确认。所有结果仍是历史、已接触研究背景下的探索性机制诊断；继承保守C的财务披露版本PIT限制。

冻结计算提交：`{freeze['commit']}`。私有运行：`{run}`。完整名单见[变量分类](results/feature_classification.csv)，逐队列全部正负结果见[置换结果](results/permutation_by_cohort.csv)，时间支持见[数据审计](DATA_ALIGNMENT_AUDIT_ZH.md)。
'''
    (public/'RESULTS_ZH.md').write_text(text)
    parts=['# 全部83项变量的诊断分类\n\n这些是候选建议，没有删除或重训。7项审计变量只得到成组证据，不能逐项声称独立贡献。证据不足默认暂时保留。\n']
    for outer in [2023,2024]:
        parts.append(f'## 截至{outer}-01-01之前的信息\n')
        for cat in ['保留','疑似冗余或噪声','证据不足']:
            sub=d[(d.outer_cutoff==outer)&(d.category==cat)]
            parts.append(f'### {cat}：{len(sub)}项\n\n'+table(sub[['feature','name','label','reason']])+'\n')
    (public/'VARIABLE_TRIAGE_ZH.md').write_text('\n'.join(parts))
    audit_text='''# 时间与数据支持核验

只读取原冻结训练快照、2019—2023历史森林和已有预测。未读取2023/2024最终测试结局。来源的sample_id、公司组、字段次序、原点、停止日、事件、日历月与原训练数组逐项核对；每个历史模型的训练随访重新截到该年1月1日之前并核对原数组，不改变模型。

全部历史训练公司与该模型的评价公司不交叉。财务/市场可得日期不晚于预测原点；原点继续采用ST公告后下一交易日开盘前，结局继续采用公告日、先退市/先恢复竞争定义。当前字段时钟仍不是原始披露版本的独立PIT认证。

原始2024训练含2023企业，345家公司26例退市。本轮新增2023队列仅从其已截断训练记录提取，不使用2023测试标签。全诊断队列2019—2023去重203家；截止2023的队列仅2019—2022共167家。二者高度重叠，不能相加当370家。

时间支持先于指标判断：无人随访到时限且仍有未知结局，则该时限指标全部NA；不能只评价早期已知事件，也不能让零权重产生看似完美的零Brier。主要筛选只用整年入组队列均已达到时限的窗口；其他结果不用于分类。

'''+table(support)+'\n\n2019模型没有训练退市事件，即使2019评价出现退市，也不能据其零置换增量删变量。2023队列只有不足9个月观察，12/24个月终点仍不受支持。\n'
    (public/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text(audit_text)
    methods='''# 可复现方法与解释边界

三个种子等权预测（每种子32树）构成每年96树固定预测器，q2、深度4、共享切点、均衡25变量、父节点AJ10均不变。128个共同企业置换在各历史入组年度内部产生，所有单变量/相关组/二元组复用同一排列，字段值与NaN一起移动；结局和随访不动。审计字段76—82使用同一企业供体联合移动。

delta_AUC/AP = 原指标 − 各次置换指标的平均；delta_Brier = 各次置换Brier平均 − 原Brier。先分别计算每次置换的指标，不把平均置换概率代入非线性AUC/AP。跨窗口汇总是对符合事件及成熟度门槛的年度效果等权平均，不是pooled AUC/AP；没有将平均年度AP称为合并AP。

1000次公司级配对bootstrap在年度内抽样，各对象/原因/时点共用抽样索引。AUC与Brier差值对全部128个固定排列先求平均；固定评价IPCW权重，主要完整队列权重实际为1。区间不包含训练样本变化、模型搜索、未来制度变化和变量筛选不确定性。无事件/无对照的重抽样记为无法计算，跨窗口平均要求各窗口该次指标都有效；未作多重检验校正，不能据区间挑选确认性显著结论。

单变量置换测量当前固定模型的依赖，相关指标会互相替代；同源相关块联合置换能补充检查，但仍可能与块外指标产生不现实组合。这不是条件独立检验，也不是因果效应。仅有低置换重要性不能认定真实无用，尤其历史退市模型仅0/2/3/6/14例训练事件。

原生函数参考分布固定为各历史训练集的乘积边际。函数精确重构仅证明忠实于当前模型；相关特征下的未观察组合、弱事件支持和交互未被学到的可能性仍保留。成对联合置换与两个单变量之和的差值只作非加性描述，不等于纯二阶函数效应。单独减去原生二元项后只评价排序分数，不宣称CIF有效或训练消融完成。

保留规则、相关阈值、事件门槛和疑似噪声条件在正式置换前冻结，详见frozen_protocol.json。尤其疑似纯噪声要求至少两个成熟窗口的训练退市均≥10例；本轮没有这样的支持，因此“不显著”不能转成“噪声”。

方法依据：[置换重要性](https://scikit-learn.org/stable/modules/permutation_importance.html)、[相关变量的置换局限](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)、[竞争风险目标及竞争结局](https://www.randomforestsrc.org/articles/competing.html)。
'''
    (public/'METHODS_ZH.md').write_text(methods)
    env=dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','scikit-learn','pytest']})
    (public/'environment.json').write_text(json.dumps(env,ensure_ascii=False,indent=2))
    receipt=dict(status='COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),private_run=str(run),code_commit=freeze['commit'],
        current_reference_sha256=REFERENCE_SHA,new_forest_fits=0,new_calibration_fits=0,reused_forests=15,successful_tasks=40,
        failed_prior_tasks=40,failed_prior_job=2582,successful_job=gate['job'],by_host=gate['by_host'],compute_seconds=gate['seconds'],
        repeats_per_object=128,company_bootstrap=1000,unique_cohort_objects=check['unique_items'],new_permutation_objects=check['unique_items']*128,
        final_tests_passed=187,full_reconstruction_error=float(val.full_reconstruction_error.max()),permutation_error=check['maximum_permutation_error'],independent_validation=ind,
        classification_counts={str(y):{k:int(v) for k,v in d[d.outer_cutoff==y].category.value_counts().items()} for y in [2023,2024]},
        current_model_unchanged=True,variables_deleted=0,test_outcomes_read=False,first_attempt_preserved=True,
        conclusion='AUDIT_BLOCK_RETAIN; REDUNDANCY_CANDIDATES_ONLY; NO_PURE_NOISE_IDENTIFIED; PREDICTIVE_GAIN_NOT_TESTED')
    (public/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    # Publish source/model hashes but no arrays, parameters or company rows.
    for n in ['source_hashes.json','immutable_input_hashes.json']:shutil.copy2(run/'audit'/n,out/n)
    hashes={str(p.relative_to(public)):sha(p) for p in public.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact_hashes.json'}
    (public/'artifact_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();report(a.run,Path(__file__).resolve().parents[2])
