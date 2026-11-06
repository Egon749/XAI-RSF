"""Aggregate-only research delivery. No company-level artifact enters Git."""
from pathlib import Path
import argparse,json,shutil,subprocess,platform,datetime
import numpy as np
import pandas as pd
from .raw_scan import sha
from .stage import successful

def table(d):return d.to_markdown(index=False)
def fmt(x,n=4):return 'NA' if pd.isna(x) else f'{x:.{n}f}'

def figures(exp,primary,ci):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['Noto Sans CJK SC','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={12:'#2563a5',24:'#c66c29'}
    for ax,metric,title in [(axs[0,0],'auc','AUC：首次财务／审计类ST排序'),(axs[0,1],'ap','AP：兼顾查准率与召回率'),(axs[1,0],'top20_coverage','前20%筛查：目标事件覆盖率')]:
        for h in [12,24]:
            d=primary[(primary.scope!='pooled')&(primary.horizon_months==h)&primary.supported].sort_values('scope')
            x=d.scope.astype(int).to_numpy();v=d[metric].to_numpy();ax.plot(x,v,'o-',label=f'{h}个月',color=colors[h],lw=2)
            for xx,vv,r in zip(x,v,d.itertuples()):
                label=f'{int(r.top20_captured)}/{int(r.target_records)}' if metric=='top20_coverage' else f'{vv:.3f}'
                offset=9 if h==12 else -18
                if metric=='ap' and xx==2024:offset=-18 if h==12 else 9
                ax.annotate(label,(xx,vv),xytext=(0,offset),textcoords='offset points',ha='center',color=colors[h],fontsize=9)
        ax.set_xticks([2023,2024,2025]);ax.grid(alpha=.2);ax.set_title(title,fontsize=12);ax.legend(frameon=False)
        ax.set_ylim((.75,1) if metric=='auc' else (0,.3) if metric=='ap' else (.5,.95))
    ax=axs[1,1];d=primary[(primary.scope!='pooled')&(primary.horizon_months==12)].sort_values('scope');x=np.arange(3)
    ax.bar(x-.18,100*d.risk_mean,.36,label='平均预测',color='#2563a5');ax.bar(x+.18,100*d.observed_aj,.36,label='观察发生率（AJ）',color='#6f8f70')
    ax.set_xticks(x,d.scope);ax.set_ylabel('12个月概率（%）');ax.set_title('概率水平：当前略有高估',fontsize=12);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_ylim(0,1.8)
    fig.suptitle('XAI-RSF：正常企业首次ST预警迁移',fontsize=16)
    fig.supxlabel('探索性 / PROVISIONAL；75项事前输入；2025年24个月随访不足，不报告性能',fontsize=10)
    out=exp/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'warning_summary.png',dpi=220);fig.savefig(out/'warning_summary.pdf');plt.close(fig)

def deliver(run):
    exp=Path(__file__).resolve().parent;repo=exp.parents[1]
    boot=successful(run,'bootstrap',7)
    assert {(v['scope'],v['horizon']) for _,v in boot}=={(s,h) for h in [12,24] for s in ['2023','2024','2025','pooled'] if (s,h)!=('2025',24)}
    ci=pd.concat([pd.read_csv(p/'BOOTSTRAP_95.csv') for p,_ in boot],ignore_index=True);ci.to_csv(run/'reports/BOOTSTRAP_95.csv',index=False)
    metrics=pd.read_csv(run/'reports/METRICS_ALL.csv');primary=metrics[(metrics.seed=='ensemble')&(metrics.version=='shrunk')&(metrics.cause==1)]
    secondary=metrics[(metrics.seed=='ensemble')&(metrics.version=='shrunk')&(metrics.cause==2)]
    main=[]
    for r in primary.itertuples():
        main.append({'年度':r.scope,'时限（月）':r.horizon_months,'公司数':int(r.records),'目标事件':fmt(r.target_records,0),'AUC':fmt(r.auc),'AP':fmt(r.ap),
            'IPCW Brier':fmt(r.ipcw_brier,6),'Top20捕获':f'{int(r.top20_captured)}/{int(r.target_records)}' if r.supported else 'NA',
            'Top25捕获':f'{int(r.top25_captured)}/{int(r.target_records)}' if r.supported else 'NA',
            '平均预测':f'{r.risk_mean:.2%}','观察AJ':f'{r.observed_aj:.2%}' if r.supported else 'NA'})
    comp=pd.read_csv(run/'reports/EXPLANATION_COMPLEXITY.csv');comp=comp[(comp.seed=='ensemble')&(comp.version=='shrunk')]
    v=pd.read_csv(run/'reports/FUNCTION_VALIDATION.csv');maxerr=v[['full_reconstruction','explicit_component_sum','function_reload','forest_reload']].to_numpy().max()
    null=pd.read_csv(run/'reports/NULL_AJ_BASELINE.csv');nc=null[null.cause==1][['scope','horizon_months','ipcw_brier']].rename(columns={'ipcw_brier':'AJ常数Brier'})
    nc=nc.merge(primary[['scope','horizon_months','ipcw_brier']],on=['scope','horizon_months']).rename(columns={'scope':'年度','horizon_months':'时限（月）','ipcw_brier':'XAI-RSF Brier'})
    perf_ci=ci[(ci.cause==1)&ci.metric.isin(['auc','ap','top20_coverage'])].copy()
    perf_ci['95%区间']=perf_ci.apply(lambda x:f"[{x.lower95:.4f}, {x.upper95:.4f}]",axis=1)
    ci_pivot=perf_ci.pivot(index=['scope','horizon_months'],columns='metric',values='95%区间').reset_index()
    result=f'''# 首次财务困境预警迁移结果

已使用本地CSMAR原始文件完成迁移并实际训练。三个年度的96树XAI-RSF均可输出合法竞争风险CIF和完整精确解释。12个月AUC依次为0.9088、0.8604、0.8521，AP依次为0.2281、0.1417、0.0985，前20%捕获29/36、33/46、41/57。24个月2023/2024的AUC为0.8601/0.8344，前20%捕获54/82、70/103。**这些是正常企业的首次财务／审计类ST预警结果，不是原ST后退市结果。** 当前平均预测略高于观察发生率，并未复现原ST后任务的总体低估。全部结果为探索性、PROVISIONAL，原始披露版本尚未独立核验，不能宣布正式PIT验证或方法创新已获得确认。

## 主要结果

先平均17/503/22三个种子的逐公司预测，再计算指标。每个种子32棵树。12个月合并2023—2025；24个月合并2023/2024。2025年24个月只导出预测与数值验证，性能为NA。

{table(pd.DataFrame(main))}

Top20/25在各年度全体预测公司中按风险排序、向上取整确定名额。未知结局不算正常或误报。2023两个时限均无未解决评价结局；2024年24个月、2025年12个月各有1条未解决结果，二元AUC/AP排除该条，IPCW Brier和AJ保留删失。完整逐年度、逐种子、raw/shrunk、两个原因结果见METRICS_ALL.csv。

24个月窗口跨年重叠，pooled的185条目标事件记录来自139家独立目标事件公司；Top20捕获124条记录，去重后捕获100/139家公司。12个月合并139条目标记录也对应139家独立公司，Top20捕获103/139。不能将跨年重叠记录当成新增独立事件。

12个月pooled AUC为0.8708，年内配对数加权AUC为0.8687；24个月分别为0.8458与0.8453。合并AP由合并预测重算，没有平均年度AP。

## 不确定性与概率水平

以下为1000次公司级重抽样的描述性95%区间，模型和预测固定。合并期抽样将同一公司的不同年度记录放在同一抽样单元中。区间不包含重新训练、种子选择、数据口径或后续调参的不确定性。

{table(ci_pivot)}

12个月平均预测为1.292%、1.318%、1.422%，观察AJ为0.865%、1.023%、1.253%。模型总体存在轻度绝对高估，尤其2023年；没有对输出做归一化、放大、截断或校准。模型的风险排序与绝对概率质量是不同问题。

与仅用对应截止日前训练结局计算的AJ常数预测相比，各个可评价的年度/时限的Brier均较低。这里只能说明当前固定数据上的概率误差比较，不代表已经优于标准RSF等尚未在新风险集上运行的对照。

{table(nc.round(6))}

## 完整解释与计算

{table(comp[['year','trees','max_path_variables','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells']])}

全部训练和评价行、所有原生时间点及两个原因的最大重构／组件求和／保存重载误差为{maxerr:.3e}，小于1e-10。CIF非负、随时间非减，原因和不超过1。原森林、raw及收缩模型均保存；每年的96树预测与三个种子预测均值一致。合并核验仅缓存固定系数以减少重复运算；普通导出器与缓存实现的结果逐样本核对，未改变数学目标或输出。

固定75个输入坐标中，各年实际非零主效应34、35、38个，二元交互77、86、86个。完整空间表约2100格；缺失状态单列，乘以时间和原因后的标量规模另报。乘积分布参考下约26%—27%的空间格没有训练联合样本，低支持格约31%—33%；精确重构并不等于每个交互区域都有充分经济或统计证据。贡献是CIF加性贡献，不是因果效应。

实际执行9次正式32树拟合，另有1次4树试运行，共292棵新训练树；3个96树合并模型由正式森林平均产生，不是新增拟合。原始扫描20项、正式拟合9项、试运行1项、合并核验3项、bootstrap任务7项均保留记录。原始扫描在oliver/lemon分配13/7项；正式拟合由动态池分配到oliver；3个合并任务在lemon完成。单模型不跨机器训练。当前参考模型和所有旧结果未改动。

## 数据限制与研究判断

本地行情最早从2015年开始，所以可核验年度训练入组实际从2016年起；不能把本次写成2012年起完整样本。输入在预测日前构造，不包含ST公告中才出现的入组警示或审计描述；与原74变量ST后模型的参数化输入不同。三套训练快照分别有20,145/24,308/28,806条记录，独立目标事件公司232/268/314家，快照互相包含，不能相加解释为独立样本。

当前财务CSV仍有历史原始版本未认证的问题；部分首次ST原因/公告时钟不能确认，采取显式删失。最近31天交易状态门槛排除了长时间停牌等无法确认正常状态的企业，可能造成样本选择限制。公司代码联系采用本地InstitutionID，未宣称全面法律实体核验。以上均限制正式外推结论。

本轮支持“固定XAI-RSF可从ST后预后迁移至正常企业首次ST预警，并保留精确解释”的可执行性结论；不支持“已是最优预警模型”“已通过严格PIT认证”“提高了原ST后退市性能”或“新增机制已被独立证实”。本轮没有重训比较模型、搜索参数、选择最优种子，也没有替换主模型。
'''
    (exp/'RESULTS_ZH.md').write_text(result)
    (exp/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# 研究判断

本轮为探索性迁移验证，不是新算法消融或确认性研究。固定q2森林在本地可核验风险集上实际完成训练、预测与精确函数导出，说明模型可以用于首次财务困境预警。其12个月AUC约0.85—0.91、Top20覆盖约72%—81%，但年度AP下降、平均风险略高估，不能只强调AUC。

解释表示的精确性保留；未证明所有函数已恢复真实经济作用，也未证明分组或交互稳定性优于新任务上的其他方法。不同任务、不同基线发生率、不同输入和不同风险集的指标不能直接作优劣比较。

严格时间外结局截断已经实施，但原始披露版本未认证、历史起点受本地行情限制、未知原因删失和入组状态筛选仍然存在。当前正式验证判断为INCONCLUSIVE；探索性工程迁移完成。需要补齐这些数据证据并在固定新协议下做公平对照后，才可决定论文主结果或模型升级。此次不自动进行下一轮搜索或替换当前ST后主模型。
''')
    (exp/'PAPER_METHODS_RESULTS_ZH.md').write_text('''# 可用于论文的迁移实验方法与结果草稿

我们将固定的内生可解释随机生存森林迁移至正常企业的首次财务困境预警。预测原点分别为2023、2024和2025年1月1日。研究对象为预测日前有可核验正常交易状态、无已知既往风险警示且行业分类为非金融的A股企业。目标事件为首次财务／审计类ST或*ST公告，其他首次ST及此前直接退市作为竞争事件；原因或时间无法解决的记录按删失保留。财务、市场与质量特征均由预测日前资料构造，共75项；财务当前版本与历史原始披露版本之间仍存在未认证差异，故实验标记为探索性。

各年度模型使用截至当年预测日前可观察的历史年度风险集，最早可验证入组为2016年。每条训练记录最多随访24个日历月，且在预测截止日前截断。同一公司可贡献多个历史年度记录，采用公司级bootstrap。模型固定为q≤2、深度4、共享分位切点、每树25变量的均衡覆盖、目标事件公司支持保护及强度10的直接父节点AJ收缩。种子17、503、22各训练32棵树，逐公司平均三个种子的CIF后评价；没有以本轮结果重新选种子或参数。

12个月目标事件AUC在三个年度分别为0.9088、0.8604和0.8521，AP分别为0.2281、0.1417和0.0985，Top20捕获29/36、33/46和41/57个事件。24个月在2023/2024年度的AUC为0.8601/0.8344；2025年24个月随访尚不充分，不报告性能。风险均值略高于观察AJ发生率。完整主效应与二元交互仍能精确重构所有训练及评价样本的原生时间点CIF。结果支持模型的任务迁移可执行性，但不能替代历史披露版本认证或新任务上的公平基线比较。
''')
    # Whitelist aggregate files only. No generic directory copy.
    names=['METRICS_ALL.csv','CALIBRATION.csv','NULL_AJ_BASELINE.csv','FUNCTION_VALIDATION.csv','COMPUTE.csv','EXPLANATION_COMPLEXITY.csv','FUNCTION_IMPORTANCE.csv','BOOTSTRAP_95.csv','COHORT_COUNTS.csv','EXCLUSIONS.csv']
    for n in names:shutil.copy2(run/'reports'/n,exp/n)
    pd.DataFrame(main).to_csv(exp/'PRIMARY_RESULTS.csv',index=False)
    figures(exp,primary,ci)
    jobs=[]
    for stage in ['raw_scan','pilot','formal','ensemble','bootstrap']:
        jobdirs=[p for p in (run/'results'/stage).iterdir() if p.name.isdigit()]
        assert len(jobdirs)==1,'Audit all retries explicitly before delivery'
        job=int(jobdirs[0].name)
        matches=list(Path('/home/oliver/.local/share/heteropool-20261005/auto-runs').glob(f'*/tasks.jsonl.results-{job}.json'));assert len(matches)==1
        d=json.loads(matches[0].read_text());assert len(d['completed'])==d['expected'] and not d['pending']
        assert all(x['returncode']==0 and x['publication_completed'] for x in d['completed'])
        shutil.copy2(matches[0],run/'audit'/f'scheduler_{job}.json')
        jobs.append(dict(stage=stage,job_id=job,tasks=d['expected'],seconds=d['seconds'],by_node={host:sum(x['host']==host for x in d['completed']) for host in {x['host'] for x in d['completed']}},scheduler_sha256=sha(matches[0]),failures=0))
    pointer=repo/'CURRENT_POST_ST_MODEL.json'
    source=subprocess.check_output(['git','show','9ffcf8e07dbbaae03cec59c1802978571d516298:CURRENT_POST_ST_MODEL.json'],cwd=repo)
    assert pointer.read_bytes()==source
    receipt=dict(status='COMPLETE_EXPLORATORY_PROVISIONAL',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branch='research/xai-rsf-pre-st-warning-v1',
        source_commit='9ffcf8e07dbbaae03cec59c1802978571d516298',protocol_commit=subprocess.check_output(['git','rev-parse','e71e258'],cwd=repo,text=True).strip(),
        compute_stages={s:json.loads((run/'audit'/f'{s}_stage.json').read_text()) for s in ['pilot','formal','ensemble','bootstrap']},
        formal_forest_fits=9,pilot_forest_fits=1,forest_fits=10,new_trees=292,formal_trees=288,calibrator_fits=0,ensemble_predictors=3,unit_test_synthetic_fits_excluded=True,
        current_pointer_unchanged=True,current_pointer_sha256=sha(pointer),seeds=[17,503,22],features=75,training_start_year=2016,primary_months=12,
        maximum_reconstruction_error=float(maxerr),all_probability_checks_pass=True,all_save_reload_checks_pass=True,jobs=jobs,
        source_raw_files=json.loads((run/'audit/raw_scan_manifest.json').read_text()),financial_sources=json.loads((run/'prepared/financial_sources.json').read_text()),
        event_source_audit=json.loads((run/'audit/event_sources.json').read_text()),
        prepared_inputs={str(p.relative_to(run/'prepared/folds')):sha(p) for p in (run/'prepared/folds').rglob('*.npz')},
        output_model_hashes=v[['year','seed','version','model_sha256','functions_sha256']].to_dict('records'),
        limitations=['Current-vintage financial export not independently PIT certified','2012-2015 origins unavailable from local daily data','Unknown cause/announcement transitions censored','Prior C039 seed selection exploratory','2025 H24 unsupported; performance NA'],
        failures=['Two initial directory creation attempts on MainDiskData waited on filesystem leases and were terminated before any fit; independent shared HDD run used instead.',
                  'Initial optional-package test skips resolved in an isolated test environment; full configured suite passed.'],
        results_commit_note='Commit containing this receipt is packaging; full result commit recorded after commit creation.')
    (exp/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,default=str))
    print(primary[['scope','horizon_months','auc','ap','ipcw_brier','top20_captured','target_records']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();deliver(a.run)
