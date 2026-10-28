"""Publish aggregate results only; no fitted parameters or company records."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,write

LABEL={'xai_q2':'XAI-RSF q2','unrestricted':'同后端无限阶','standard_rsf':'标准RSF',
       'training_AJ':'训练AJ','calibration_AJ':'校准期AJ'}


def table(df):
    def cell(v):
        if pd.isna(v):return 'NA'
        return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    rows=[list(df.columns),['---']*len(df.columns),*[[cell(v) for v in row] for row in df.itertuples(index=False,name=None)]]
    return '\n'.join('| '+' | '.join(row)+' |' for row in rows)+'\n'


def figures(df,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,'svg.hashsalt':'affine23-24'})
    for metric,title in [('brier','12个月Brier：越低越好'),('risk_mean','平均预测风险与实际发生比例')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8));models=['xai_q2','unrestricted','standard_rsf']
        for row,year in enumerate([2023,2024]):
            for col,cause in enumerate(['termination','recovery']):
                ax=axes[row,col];part=df[(df.test_year==year)&(df.cause==cause)];x=np.arange(3)
                for shift,state,color,label in [(-.18,'uncalibrated','#b6bdc5','未校准'),(.18,'affine','#3278aa','仿射校准')]:
                    d=part[(part.seed=='ensemble')&(part.state==state)].set_index('model').loc[models]
                    bars=ax.bar(x+shift,d[metric],width=.34,color=color,label=label)
                    ax.bar_label(bars,labels=[f'{v:.3f}' for v in d[metric]],padding=3,fontsize=9)
                base=part[part.model=='calibration_AJ'].iloc[0]
                line=base.brier if metric=='brier' else base.ipcw_observed_incidence
                ax.axhline(line,color='#b34b3d',linestyle='--',label='校准期AJ基线' if metric=='brier' else '测试期实际发生比例')
                ax.set_ylim(0,max(part.loc[part.seed=='ensemble',metric].max(),line)*1.3+.005)
                ax.set_xticks(x,[LABEL[m] for m in models]);ax.spines[['top','right']].set_visible(False)
                ax.set_title(f'{year}年 · '+('退市' if cause=='termination' else '恢复')+f'（{int(base.known_target_events)}例）')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=10,frameon=False)
        fig.suptitle(title,fontsize=16)
        fig.text(.5,.015,'固定三个种子先平均预测，再拟合公共校准器；校准仅使用测试年份之前的信息。\n2023年36家、2024年45家；时间外探索性比较，无显著性或独立盲测声明。',ha='center',fontsize=10)
        fig.tight_layout(rect=(0,.065,1,.9));name='brier_comparison' if metric=='brier' else 'risk_level_comparison'
        fig.savefig(out/f'{name}.png',dpi=170);fig.savefig(out/f'{name}.svg',metadata={'Date':None});plt.close(fig)
        p=out/f'{name}.svg';p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')


def deliver(run,destination):
    valid=json.loads((run/'evaluation/independent_validation.json').read_text());assert valid['status']=='PASS'
    out=destination/'results';out.mkdir(exist_ok=False)
    for folder in ['evaluation','reports']:
        for p in (run/folder).iterdir():
            assert p.suffix in ['.csv','.json'] and 'PRIVATE' not in p.name
            shutil.copy2(p,out/p.name)
    for name in ['data_alignment.json','attempt_history.json','environment_oliver.json','environment_lemon.json',
                 'pilot_gate.json','formal_gate.json','tests_final.log','pre_fit_freeze.json']:
        shutil.copy2(run/'audit'/name,out/name)
    shutil.copy2(run/'calibration/freeze.json',out/'calibration_freeze.json')
    assert '91 passed' in (out/'tests_final.log').read_text() and ' skipped' not in (out/'tests_final.log').read_text()
    m=pd.read_csv(out/'metrics.csv',dtype={'seed':str});w=pd.read_csv(out/'warning_metrics.csv',dtype={'seed':str})
    c=pd.read_csv(out/'cohort_counts.csv');diagnostics=pd.read_csv(out/'training_diagnostics.csv')
    complexity=pd.read_csv(out/'interpretability.csv');solver=pd.read_csv(out/'calibration_solver.csv',dtype={'seed':str})
    figures(m,out)
    def doc(name,text):
        with (destination/name).open('x') as f:f.write(text.strip()+'\n')
    def metrics(cause):
        d=m[(m.cause==cause)&m.seed.isin(['ensemble','none'])].copy();d['model']=d.model.map(LABEL)
        return table(d[['test_year','model','state','known_target_events','auc','ap','brier','risk_mean','ipcw_observed_incidence']])
    doc('RESULTS_ZH.md','''# 2023／2024仿射再校准结果

**实际完成18个森林、24个校准器、2个4树试运行。两年分别评价，观察期均为12个日历月。** 主要结果为固定三个种子预测先平均，再拟合一个公共仿射校准器；不是各自校准后再平均，也不是三个指标简单平均。逐种子全部结果另存CSV。

## 退市

'''+metrics('termination')+'''

q2退市Brier：2023由0.109322降至0.106667；2024由0.319896降至0.294980。AUC/AP和Top20/25名单完全不变。2023 q2退市斜率触及预先规定的正数下界，概率差异接近消失；数学排序保留不表示仍有足够可用的概率分辨幅度。其Brier与仅使用近期发生水平的校准期AJ几乎相同。2024 q2略低于校准期AJ的0.297097，但没有不确定性证据支持稳定优越性。

2024 q2退市平均风险由1.73%升至5.97%，实际15/45=33.33%；历史校准没有解决大幅低估。2023实际4/36=11.11%，校准后2.22%，仍低估。校准器看不到未来制度的真实发生率，不能把概率约束或结构保持等同于消除漂移。

2023标准RSF的AUC为0.9453，但只有4个退市事件，不能据此宣称稳定高精度。所有模型排序数值均来自本轮基础森林，不是仿射校准带来的AUC增益。

## 恢复

'''+metrics('recovery')+'''

2023没有12个月内先恢复事件，恢复AUC/AP为NA，不能填0。2024只有4例恢复，q2 AUC0.4146，恢复区分度仍弱。q2恢复Brier在两年均下降，但两年都高于校准期无变量AJ。恢复的平均预测从29.63%降到10.79%（实际0%），从23.33%降到14.32%（实际8.89%）。

24个校准器的两原因测试Brier均有所下降，包含全部18个单种子与6个集成组合；这是一轮探索性观察，不是保证未来总会改善，也没有显著性声明。使用联合两原因Brier拟合校准，未按测试结果挑选校准版本。

![Brier前后对照](results/brier_comparison.png)

![平均风险前后对照](results/risk_level_comparison.png)

本轮为留出独立校准集而使用更早、更小的森林训练集；不能将本轮AUC与此前全训练集／24个月或2025实验直接比较，归因于仿射校准或模型机制改进。2023/2024历史结局已被研究接触，状态为PROVISIONAL_EXPLORATORY_TEMPORAL_CALIBRATION。
''')
    ww=w[(w.seed=='ensemble')&(w.state=='affine')].copy();ww['model']=ww.model.map(LABEL)
    doc('PREDICTION_COMPARISON_ZH.md','''# 排序与核查覆盖

同一结局、同一年份和同一校准版本下，正斜率仿射变换保持全部公司的排序和并列关系。本轮逐公司核验、独立指标复算均通过。

'''+table(ww[['test_year','model','top_fraction','checked','captured','target_events','coverage','known_non_target','unknown','lead3','lead6']])+'''

q2：2023 Top20/25分别核查8/9家，都捕获1/4例；2024核查9/12家，分别捕获5/15及6/15例。校准前后相同，不能宣传事件覆盖提高。已知非目标包含恢复和时限内未发生首次转归，不能全称为正常；未知0，一家公司一条记录，无重复预警。

这是全年入组池的回顾性核查容量对照，不是可提前知道全年公司池的在线策略。提前3/6个月从预测起点到首次退市公告按日历月计算。本轮捕获的事件无一满足至少6个月提前量。完整校准分箱见CSV，仅用于描述，不据此修改参数。
''')
    doc('INTERPRETABILITY_COMPARISON_ZH.md','''# 精确解释与概率约束

更新为 F'_k(t|x)=c_k g_k(t)+b_k F_k(t|x)。基准变为 c_k g_k(t)+b_k mu_k(t)，每个主效应及二阶交互乘以同一个正数b_k。固定训练参考分布，没有新增变量、空间切点、高阶残差或后处理裁剪。

'''+table(complexity)+f'''

逐种子以及公共校准后的q2集成均核验了完整函数重构，最大误差{valid['q2_reconstruction_max_error']:.3e}。相对平均绝对贡献占比保持不变；全部变量/交互集合和跨种子集合稳定性也保持不变，不能据此声称解释稳定性提高。2023 q2集成退市分量缩放到接近零是重要限制，形式上保留的函数不能被描述成具有强实际贡献。

g来自森林训练数据的AJ曲线，按第365天值归一化并于365天后保持1；g(0)=0且不减。c>=0、b>=1e-6、sum(c)+max(b)<=1为整个原始概率单纯形提供有效性保证，因而不依赖仅检查已有公司。数值可行性误差上限1e-10。所有实际曲线另验起点0、单调、非负及原因和<=1。只在12个月拟合校准；中间时点曲线形状是预先规定的，不宣称已在3/6个月校准。

这是RSF输出后的显式仿射更新层，保留基础森林的bootstrap、分裂及叶节点估计机制，但更新后的CIF不能再称为原始未更新森林的输出。贡献仍在CIF概率尺度上，不是风险率贡献或因果效应。校准参数、完整函数与公司解释保留服务器。
''')
    doc('DATA_ALIGNMENT_AUDIT_ZH.md','''# 三段时间对齐

'''+table(c)+'''

2023基础森林在2021-01-01前冻结信息；校准用2021/2022入组，信息截止2023-01-01前；测试2023。2024基础森林在2022-01-01前冻结信息；校准用2022/2023入组，信息截止2024-01-01前；测试2024。每段自身随访均截断在12个月。校准与测试公司没有进入对应森林训练，三组公司两两不重合。2023作为前一实验测试及后一实验历史校准仅使用后一时点已经可得部分，属于预先规定的滚动协议。

独立日期逻辑核验735条分段记录，含跨折重复；基础训练独立公司最多277家，不能把两折重复相加当作独立样本。测试两年共81家，19个先退市、4个先恢复，未知0。2023校准仅1例退市、4例恢复，2024仅4例退市、5例恢复，均有32条早期删失；校准使用日历月时间上的reverse-KM IPCW，未知端点权重0，从不当成阴性。缺少条件删失敏感性分析，不能主张完全校正所有删失偏差。

75个入组特征逐值复用冻结数据，财务和市场可得日仍严格早于公告日起点，未重新选择企业。2023/2024随访恢复与原24个月审计的首次A/X顺序一致，再截到12个月。终点仍是首次完整恢复A或最终终止上市X，T及中间警示变化不是终点。数据源继承覆盖到2026-09-03，不以今天代替随访证据。

未使用2025入组评价，也未添加规则日历或政策变量。原始财报版本、行业时点和部分生命周期首次历史证据未完全认证，同日年报仍被旧公告日起点排除。本轮检验的是仿射校准兼容性及时间外结果，不代表这些上游问题已经修复。所有旧实验保持原样。
''')
    ss=solver[solver.seed=='ensemble']
    doc('TRAINING_DIAGNOSTICS_ZH.md','''# 执行与求解器记录

18/18正式森林、2/2试运行、24/24校准器完成，计算及结果回传核验成功。Slurm作业2540、2541申请两节点physical池；两机Python和共享路径通过预检，实际20个短拟合均由oliver领取，lemon没有拟合，不能宣称双机实测加速。每任务单线程、4GiB地址空间限额，没有GPU计算。

'''+table(diagnostics[['test_year','model','seed','training_companies','training_termination_events','training_recovery_events','seconds','peak_rss_MiB','child_peak_rss_MiB','reload_error']])+'''

基础森林每模型32树，深度6，所有原配置和原因权重(1,1)不变。完整测试91项通过，0跳过，含独立R检查。唯一警告为既有EBM缺失值可视化提示。

## 校准求解

'''+table(ss[['test_year','model','calibration_companies','known','unknown','termination','recovery','objective_before','objective_after','gap','primal_violation','slope_floor_active','simplex_constraint_active']])+f'''

联合IPCW Brier为4参数凸二次目标，枚举最多4个活动不等式的57个KKT系统；表内iterations=57表示检查的活动集数量，不是森林迭代次数。独立线性规划计算Frank–Wolfe gap，最大{valid['solver_max_gap']:.3e}，小于预先阈值1e-8；概率约束误差小于1e-10。全部单种子结果在CSV，不选最好种子。

曾有1项合成测试失败：SLSQP声明成功但gap为7.88e-7，未达到1e-8。失败发生于任何森林训练之前，原代码和日志留在独立旧目录。改为KKT活动集枚举，目标、约束及检验阈值不变，随后完整91项通过；没有重跑或隐瞒失败真实拟合。真实拟合失败0，发布失败0。

104组测试指标独立复算，全部任务ID/计数、输入589个文件及模型/校准器哈希通过。24个校准器先写入冻结回执，然后才执行测试评价；无测试集选参、无再校准后重用测试集拟合。
''')
    doc('METHOD_INNOVATION_DECISION_ZH.md','''# 本轮判断

**INCONCLUSIVE：结构保持与数值实现通过，时间外概率误差有所下降；但缺少证据证明该校准模块具有稳定、独立的预测优势或解决政策漂移。**

- 已支持：在所声明全概率空间约束下，仿射更新保持合法CIF、完整二阶解释和排序。真实数据上的两年、三模型、全部种子均执行验证。
- 已观察：两原因Brier在本轮所有匹配组合中下降，概率水平向观察值靠近；不是AUC/AP或核查覆盖提高。
- 未解决：2024退市发生率33.33%，q2校准后平均仅5.97%；未来水平变化仍远超此前校准数据。
- 独立价值有限：2023 q2退市几乎退化成校准期AJ常数预测；2024退市仅略优于该基线；恢复两年均不如校准期AJ。不能把“比失准原模型好”当成“比简单近期发生率好”。
- 解释稳定性没有提高：函数集合及相对贡献保持不变；概率幅度缩小不是减少解释歧义或提高跨种子稳定性的证据。
- 方法性质与创新分开：仿射闭包是可写入论文的结构保证，不能单凭这一代数性质宣称独立方法创新；真实收益仍须更充分的时间外证据。

两年均为已接触结局后的探索性比较；校准事件只有1/4个退市，未给出显著性结论。保留全部失败测试和不利限制，不自动扩大窗口搜索或启动其他训练。
''')
    with (destination/'README.md').open('a') as f:f.write('''
## 实际结果

18个森林、24个校准器、2个试运行完成，91项测试通过。q2集成退市Brier：2023为0.109322→0.106667；2024为0.319896→0.294980；AUC/AP、Top20/25名单不变。最大函数重构误差1.67e-16。2023退市输出接近常数，2024平均风险仍明显低估。判断INCONCLUSIVE。

- [完整结果](RESULTS_ZH.md)、[事件核查](PREDICTION_COMPARISON_ZH.md)、[解释与概率约束](INTERPRETABILITY_COMPARISON_ZH.md)。
- [数据审计](DATA_ALIGNMENT_AUDIT_ZH.md)、[数值与执行诊断](TRAINING_DIAGNOSTICS_ZH.md)、[研究判断](METHOD_INNOVATION_DECISION_ZH.md)。

## 复现入口

创建唯一私有RUN及 input/audit/reports/private/results/evaluation_input 空目录；不要指向既有运行。使用已安装的隔离环境，先运行完整pytest（含独立R检查）。

```bash
python -m experiments.post_st_affine_calibration_v1.prepare --run "$RUN"
python -m experiments.post_st_affine_calibration_v1.stages --run "$RUN" --repo "$PWD" --stage freeze
/home/oliver/.local/bin/hetero-submit "$RUN/pilot.jsonl" physical both
python -m experiments.post_st_affine_calibration_v1.stages --run "$RUN" --stage pilot --job "$PILOT_JOB"
/home/oliver/.local/bin/hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_affine_calibration_v1.stages --run "$RUN" --stage formal --job "$FORMAL_JOB"
python -m experiments.post_st_affine_calibration_v1.analyze --run "$RUN" --stage calibrate
python -m experiments.post_st_affine_calibration_v1.analyze --run "$RUN" --stage evaluate
python -m experiments.post_st_affine_calibration_v1.validate --run "$RUN"
```

训练前必须冻结代码；只在试运行和正式阶段门检通过后进入下一步。所有输出唯一创建，禁止覆盖历史目录。环境版本、模型及输入哈希见results与execution_receipt.json。企业明细、校准系数、预测及原函数参数留在服务器。
''')
    freeze=json.loads((run/'calibration/freeze.json').read_text())
    receipt=dict(experiment='post_st_affine_calibration_v1',status='COMPLETE',decision='INCONCLUSIVE',
        data_status='PROVISIONAL_EXPLORATORY_TEMPORAL_CALIBRATION',private_run=str(run),
        training_commit=str(diagnostics.code_commit.iloc[0]),branch='research/xai-rsf-post-st-prognosis-v1',
        formal_fits=18,pilots=2,calibration_fits=24,failed_fits=0,failed_publications=0,
        failed_pre_fit_tests_preserved=1,final_full_tests_passed=91,final_full_tests_skipped=0,
        training_rows_with_fits_and_seed_repetition=int(diagnostics.training_companies.sum()),
        unique_forest_training_companies=277,test_companies=81,test_termination_events=19,test_recovery_events=4,
        test_years=[2023,2024],horizon_calendar_months=12,cohort_counts=c.to_dict('records'),
        validation=valid,calibration_hashes=freeze['hashes'],test_outcome_calibration=False,
        no_new_rule_features=True,no_cause_weight_change=True,no_2025_evaluation=True,
        prior_experiments_preserved=True,privacy='No company data, predictions, thresholds, fitted coefficients or function values published',
        jobs={k:json.loads((run/f'audit/{k}_gate.json').read_text()) for k in ['pilot','formal']},
        source_hashes=json.loads((run/'audit/data_alignment.json').read_text())['source_sha256'],
        results_sha256={p.name:sha(p) for p in sorted(out.iterdir())},
        documents_sha256={p.name:sha(p) for p in sorted(destination.glob('*.md'))},
        code_sha256={p.name:sha(p) for p in sorted(destination.iterdir()) if p.suffix in ['.py','.json']})
    write(destination/'execution_receipt.json',receipt);shutil.copy2(destination/'execution_receipt.json',run/'execution_receipt.json')
    print(json.dumps(dict(status='PACKAGED',public_files=len(receipt['results_sha256']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();deliver(a.run,a.destination)
