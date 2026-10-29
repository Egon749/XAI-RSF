"""Publish aggregate evidence, never company records or fitted function values."""
from pathlib import Path
import argparse,datetime,json,re,shutil
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,write
from experiments.post_st_next_open_info_v1.deliver import table


def deliver(run,dest):
    alignment=json.loads((run/'audit/data_alignment.json').read_text())
    for n,h in alignment['unchanged_source_hashes'].items():assert sha(n)==h
    for n,h in json.loads((run/'audit/immutable_input_hashes.json').read_text()).items():assert sha(run/'input'/n)==h
    val=json.loads((run/'evaluation/validation.json').read_text());ind=json.loads((run/'evaluation/independent_audit.json').read_text())
    jobs={s:json.loads((run/f'audit/{s}_gate.json').read_text()) for s in ['pilot','formal']}
    assert all(z['status']=='PASS' for z in [alignment,val,ind,*jobs.values()])
    plan=json.loads((run/'input/folds/2025/plan.json').read_text())
    protocol=json.loads((run/'audit/protocol.json').read_text())
    tasks=[]
    for stage,gate in jobs.items():
        manifest=[json.loads(x) for x in (run/f'{stage}.jsonl').read_text().splitlines()]
        for i,item in enumerate(manifest):
            argv=item['argv'];model=argv[argv.index('--model')+1];seed=int(argv[argv.index('--seed')+1])
            folder=run/'results'/stage/str(gate['job'])/f'task-{i}'/'output'
            for n,h in json.loads((folder/'hashes.json').read_text()).items():assert sha(folder/n)==h
            d=json.loads((folder/'summary.json').read_text())
            tasks.append(dict(stage=stage,task=i,model=model,seed=seed,host=d['host'],seconds=d['seconds'],
                cpu_seconds=d['cpu_seconds'],peak_rss_MiB=d['peak_rss_MiB'],child_peak_rss_MiB=d.get('child_peak_rss_MiB',0),
                new_forest_fits=d.get('new_forest_fits',1),leaf_updates=d.get('leaf_updates',0),status=d['status']))
    runtime=pd.DataFrame(tasks);runtime.to_csv(run/'evaluation/execution_tasks.csv',index=False)
    logs={n:(run/'audit'/n).read_text() for n in ['full_tests_pre_fit.log','full_tests_with_R.log','tests_final.log']}
    passed=int(re.findall(r'(\d+) passed',logs['tests_final.log'])[-1]);assert passed>=147 and ' skipped' not in logs['tests_final.log'] and ' failed' not in logs['tests_final.log']
    out=dest/'results';out.mkdir(exist_ok=False)
    for p in (run/'evaluation').iterdir():
        assert p.suffix in ['.csv','.json'] and 'PRIVATE' not in p.name;shutil.copy2(p,out/p.name)
    for p in (run/'reports').glob('*.csv'):shutil.copy2(p,out/p.name)
    for n in ['data_alignment.json','pre_fit_freeze.json','pilot_gate.json','formal_gate.json',
              'environment_oliver.json','environment_lemon.json','full_tests_pre_fit.log','full_tests_with_R.log','tests_final.log']:
        shutil.copy2(run/'audit'/n,out/n)
    write(out/'frozen_model_configurations.json',{k:plan[k] for k in ['forest_common','forest_variants','standard_rsf','code_commit']})
    m=pd.read_csv(out/'metrics.csv',dtype={'seed':str});w=pd.read_csv(out/'warning_metrics.csv',dtype={'seed':str})
    main=m[(m.seed=='ensemble')&(m.horizon_months==12)&(m.cause=='termination')]
    rec=m[(m.seed=='ensemble')&(m.horizon_months==12)&(m.cause=='recovery')]
    cols=['model','auc','ap','brier','risk_mean','ipcw_observed_incidence','mean_risk_minus_observed']
    wt=w[w.seed=='ensemble'][['model','top_fraction','checked','captured','all_target_events','coverage','known_non_target','unknown','lead3','lead6']]
    cc=pd.read_csv(out/'ensemble_complexity.csv');nodes=pd.read_csv(out/'node_audit.csv');st=pd.read_csv(out/'seed_stability.csv')
    fr=runtime[runtime.stage=='formal'];tests=logs['tests_final.log'].splitlines()[-1]
    q=main.set_index('model').loc['xai_q2'];r=main.set_index('model').loc['standard_rsf']
    delta={k:float(q[k]-r[k]) for k in ['auc','ap','brier']}
    (dest/'RESULTS_ZH.md').write_text(f'''# 2025年固定配置实验完成

当前q2深度4＋直接父节点AJ收缩10已实际重训，使用83个保守C输入。训练2012—2024入组，信息严格早于2025-01-01，训练随访上限仍为24个日历月；预测2025年53家企业，结局截止2026-06-30。正式6/6拟合、2/2小试成功，无额外搜索或校准。

## 12个月先退市

'''+table(main[cols])+f'''
先平均三个种子的预测，再计算指标。相对同口径普通RSF，q2的AUC差为{delta['auc']:.6f}、AP差为{delta['ap']:.6f}、Brier差为{delta['brier']:.6f}。本次退市排序和固定核查覆盖没有超过RSF，Brier略低；不能把很小的Brier差写成已经证明优越。

'''+table(wt)+'''
前20%向上取整核查11家，前25%核查14家。53家中12个月内退市5家、先恢复3家、其余45家在12个月内无首次终点，未知0。全年度回顾排序不是实时核查政策。只有5例退市，捕获数每变动1例，覆盖率就变动20个百分点。

## 12个月先恢复

'''+table(rec[cols])+'''
恢复只有3例，指标更不稳定。两个结局的预测水平与观察频率均单独报告，不仅看AUC。

## 24个月不能在该截止日完整评价

截止2026-06-30累计先退市6家、先恢复22家、25家删失；无人达到24个月。因此24个月AUC、AP、Brier、校准和终点覆盖不计算，不能把25家当成未退市。24个月预测保存于服务器，但不是已验证的24个月结果。

本轮为已暴露数据上的固定配置迁移，PROVISIONAL。旧2025实验的0.533不是本模型结果；新旧差异同时涉及75→83输入、预测起点、训练随访12→24个月、深度和收缩，不能把差异全归于收缩。
''')
    (dest/'DATA_ALIGNMENT_AUDIT_ZH.md').write_text('''# 保守C输入与时间截断

'''+table(pd.read_csv(out/'cohort_counts.csv'))+'''
原有443条公告日起点的财务及市场向量全部复现，390条历史下一交易日起点的83维C向量逐值完全一致，包含缺失值及顺序。2025年53家使用同一函数扩展，保留原非金融行业／首次事件审计资格，不按结局排除公司。

预测起点为入组公告后下一交易日开盘前；财务可得日保留原公告及修订日期的最大值，增长／变化项同时检查前期组成数据。市场使用实际交易日和起点前最后可用日，警示类型／审计意见仅来自入组公告。没有将研究者接受年报版本的另一口径混入。

训练390家，最多24个月且严格截至2024-12-31：49例先退市、211例先恢复；另外130家未观察到首次终点，其中87家完成24个月、43家提前删失。12个月训练端点只有23例退市，不能混用49例作12个月阳性数。

2025年53家每家公司一条记录，训练／测试公司组无重叠。测试结局放在evaluation_input，拟合目录的测试文件仅有X和sample_id。所有原始来源、冻结输入、模型和回传文件核对SHA256；没有扩展权威源原2026-09-03覆盖边界，而是向前截断至用户指定日期。

保守可得日过滤仍不等于原始披露版本已独立PIT认证；历史行业版本、观察期首次定义边界及数据暴露问题保留。来源与字段审计详见data_alignment.json，企业明细不上传。
''')
    (dest/'TRAINING_DIAGNOSTICS_ZH.md').write_text(f'''# 执行和数值核验

实际拟合代码：{plan['code_commit']}。最终完整测试：{tests}。第一次测试未配置独立R入口，142通过、5跳过；补齐既有R环境后训练前147项全部通过，0跳过，原日志均保留。既有EBM缺失值绘图提示不是本轮模型错误。

两节点Slurm均申请并检查。小试作业{jobs['pilot']['job']}，实际主机{jobs['pilot']['by_host']}；正式作业{jobs['formal']['job']}，实际主机{jobs['formal']['by_host']}。任务为单线程、4GiB进程上限，由现有cgroup／调度控制亲和，未绕过P核第二线程限制，也未为增加占用率重复训练。

'''+table(runtime)+f'''
正式6次拟合（q2三个、RSF三个），另3次固定叶估计更新；小试2次4树拟合及1次叶更新。正式重复训练计算记录为390×6=2340条，独立训练公司仍390家，独立先退市事件49个。正式任务全部计算、回传和哈希验证通过，失败0、重训重试0。

一次回执读取检查误以为hetero-run报告在原清单旁，出现FileNotFoundError；实际计算已成功。已增加显式--report并逐项验证任务身份后复制回执，无拟合重跑，无模型算法变化。冻结模型代码仍为上述SHA，交付代码包含此编排修正。

独立重放{ind['nodes_checked']}个节点、{ind['leaf_estimators_checked']}个叶估计，直接风险集递推AJ和父／叶凸组合最大差{ind['max_independent_leaf_error']:.3e}。三个RSF公司bootstrap和训练中位数均核对。保存重载、CIF非负／单调／两原因和≤1全部通过，没有裁剪预测或放宽门槛。森林没有梯度优化器，Hessian和数值gap不适用。
''')
    (dest/'INTERPRETABILITY_COMPARISON_ZH.md').write_text(f'''# 完整低阶函数保留，稳定性不能由精确性推出

三个32树模型的平均等价于完整96树的代数合并，未额外拟合。所有443条训练／测试记录、全部{ind['native_and_horizon_time_points']}个原生事件及显示时点均重构并重载核验，最大误差{ind['all_time_ensemble_reconstruction_error']:.3e}。

'''+table(cc)+'''
函数是各原因CIF的主效应与二元交互贡献，全部相加还原预测；不是风险率贡献，也不是因果影响。q2限制在训练时执行，父节点混合不改变分裂几何，但精确分解不是本方法独有的性质。

'''+table(st)+'''
这里只检验三个种子的函数选择重合，不等于证明形状稳定、统计可辨识性或参数可靠。大量主效应和交互仍可能难以阅读。

'''+table(nodes)+'''
没有本叶退市事件的叶可以通过父节点借入正概率，但本叶事件支持仍为零。完整函数、切点、模型及逐企业贡献保留在私有ensemble_private和任务目录。
''')
    (dest/'PREDICTION_COMPARISON_ZH.md').write_text('''# 同口径配置比较及限制

'''+table(main[cols])+'''
两个模型共享390家训练、53家评价、83输入、训练信息截止、测试截止、32树／种子及公司bootstrap。普通RSF保持此前depth6、训练中位数填补、随机局部切点；q2保持depth4、共享切点、每树25变量候选和父节点收缩。这是完整既定配置比较，不是单独q或收缩的因果消融，不是首次ST预警M2。

'''+table(pd.read_csv(out/'seed_summary.csv'))+'''
单种子指标范围不是置信区间，集成指标是对平均预测重算，不是上表均值。全部种子均保留。12个月测试端点完整，IPCW权重全为1，公司等权与记录等权一致。

'''+table(pd.read_csv(out/'calibration_bins.csv'))+'''
四分组校准只描述现有预测，没有用2025标签做概率映射。5例退市不能支持反复筛选配置；2025已暴露，不能称为新的严格盲测。
''')
    (dest/'METHOD_INNOVATION_DECISION_ZH.md').write_text('''# INCONCLUSIVE

实际执行成功：当前固定深度4＋收缩配置能够在新的2025输入上训练、预测并保持完整精确函数解释。

本次12个月退市AUC0.700、AP0.181317、Top20捕获2/5；同口径RSF为0.733333、0.223413、3/5。q2没有超过RSF的排序与捕获，Brier略低；绝对退市概率仍低于观察发生率。只有5个目标事件，不能据此认定稳定优越或稳定无效。

完整96树仍有64个主效应、147个二元交互、3194个空间表格。精确性已核验，简单性和选择稳定性仍有限；没有新增证据支持newRSF2可辨识性惩罚的独立贡献，本森林也未使用该惩罚。

2025不是未接触的独立测试集；原始版本和行业时钟边界仍为PROVISIONAL。24个月结局没有足够随访支持，不发布伪完整性能。既有2023/2024结果保留；本轮结束，不增加深度、种子、时间权重、校准或超参数搜索。
''')
    receipt=dict(status='COMPLETE',research_decision='INCONCLUSIVE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        private_run=str(run),repository='Egon749/XAI-RSF',branch='research/xai-rsf-post-st-prognosis-v1',
        fitting_commit=plan['code_commit'],formal_fits=6,pilot_fits=2,formal_leaf_updates=3,pilot_leaf_updates=1,
        failed_fits=0,failed_publications=0,fit_retries=0,postprocessing_report_path_probe_failures=1,
        tests_passed=passed,tests_skipped=0,training_companies=390,training_24m_termination_events=49,
        training_12m_termination_events=23,training_24m_recovery_events=211,test_companies=53,
        test_12m_termination_events=5,test_12m_recovery_events=3,test_12m_unknown=0,
        test_24m_observed_termination_events=6,test_24m_observed_recovery_events=22,test_24m_unknown=25,
        test_observation_cutoff_inclusive='2026-06-30',supported_evaluation_months=[12],unsupported_evaluation_months=[24],
        input_columns=83,shrinkage_strength=10,max_depth=4,path_budget=2,calibration_fits=0,additional_search_trials=0,
        actual_hosts={s:j['by_host'] for s,j in jobs.items()},slurm_jobs={s:j['job'] for s,j in jobs.items()},
        maximum_full_function_reconstruction_error=ind['all_time_ensemble_reconstruction_error'],
        existing_reference_unchanged=True,strict_blind_test_claim=False,
        result_sha256={str(p.relative_to(dest)):sha(p) for p in out.iterdir() if p.is_file()},
        code_sha256={p.name:sha(p) for p in dest.glob('*.py')},document_sha256={p.name:sha(p) for p in dest.glob('*.md')},
        protocol_sha256=sha(dest/'frozen_protocol.json'))
    write(dest/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','formal_fits','pilot_fits','tests_passed','research_decision']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.dest)
