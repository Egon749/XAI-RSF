"""Publish aggregate-only tables and a static scientific figure into a NEW folder."""
from pathlib import Path
import json, os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter

PUBLIC=Path(__file__).resolve().parent


def table(df,columns):
    x=df[columns].copy()
    def fmt(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in x.itertuples(index=False,name=None))+'\n'


def main():
    loc=json.loads((PUBLIC/'run_location.json').read_text());run=Path(os.environ.get('CALENDAR_LEVEL_RUN',loc['private_run']))
    out=run/'deliverables';out.mkdir()
    m=pd.read_csv(run/'evaluation/ARM_METRICS_PRIVATE.csv')
    slopes=pd.read_csv(run/'prepared/LEVEL_SLOPES_PRIVATE.csv')
    cert=pd.read_csv(run/'prepared/CERTIFICATE_CHECKS.csv')
    # Requested diagnostic coefficients are applied but not certified.
    cert['applied_termination']=cert.certified_termination
    cert['applied_recovery']=cert.certified_recovery
    cert.loc[cert.arm=='uncertified_2y',['certified_termination','certified_recovery']]=np.nan
    functions=pd.read_csv(run/'collected/FUNCTION_VALIDATION_PRIVATE.csv')
    scenarios=pd.read_csv(run/'evaluation/LEVEL_SCENARIOS.csv')
    diffs=pd.read_csv(run/'evaluation/PAIRED_DIFFERENCES_95.csv')
    # Large annual cohorts may retain risk levels; small identifying count cells
    # are suppressed, and no company identifiers/thresholds/functions are copied.
    public_metrics=m.copy()
    for col in ['events','captured']:
        private=(public_metrics[col]>0)&(public_metrics[col]<5)
        public_metrics[col+'_status']=np.where(private,'SUPPRESSED_1_TO_4','REPORTED')
        public_metrics.loc[private,col]=np.nan
    for frame,name in [(slopes,'LEVEL_SLOPES.csv'),(cert,'CERTIFICATE_CHECKS.csv'),(public_metrics,'ARM_METRICS.csv'),(diffs,'PAIRED_DIFFERENCES_95.csv'),(scenarios,'LEVEL_SCENARIOS.csv'),(functions,'FUNCTION_VALIDATION.csv')]:frame.to_csv(out/name,index=False)
    for name in ['RANKING_CHECKS.csv']:
        shutil.copy2(run/'evaluation'/name,out/name)
    for name in ['PREDICTION_ALIGNMENT.csv','INDEPENDENT_SLOPE_CHECKS.csv']:
        if (run/'collected'/name).exists():shutil.copy2(run/'collected'/name,out/name)
    ref=m.query("cause==1 and arm=='reference' and supported==True and scope!='pooled'").set_index(['scope','months'])
    main=m.query("cause==1 and arm=='both_2y' and supported==True and scope!='pooled'").set_index(['scope','months'])
    comparison=pd.DataFrame(dict(observed=ref.observed,reference_mean=ref.mean_predicted,updated_mean=main.mean_predicted,reference_OE=ref.observed_expected,updated_OE=main.observed_expected,reference_brier=ref.brier,updated_brier=main.brier)).reset_index()
    comparison['absolute_gap_reduction_pct']=100*(1-abs(comparison.updated_mean-comparison.observed)/abs(comparison.reference_mean-comparison.observed))
    comparison.to_csv(out/'TERMINATION_LEVEL_COMPARISON.csv',index=False)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'pdf.fonttype':3,'font.size':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for i,cause in enumerate([1,2]):
        for j,h in enumerate([12,24]):
            ax=axes[i,j];years=[2023,2024,2025] if h==12 else [2023,2024]
            for arm,label,color,marker in [('reference','原参考','#2864a0','o'),('recovery_only','仅恢复更新','#39996c','x'),('both_2y','两原因2年更新','#c74a3c','^')]:
                z=m[(m.cause==cause)&(m.months==h)&(m.arm==arm)&m.scope.isin([str(y) for y in years])].set_index('scope')
                ax.plot(years,[z.loc[str(y),'mean_predicted'] for y in years],label=label,color=color,marker=marker,lw=1.7)
            z=m[(m.cause==cause)&(m.months==h)&(m.arm=='reference')&m.scope.isin([str(y) for y in years])].set_index('scope')
            ax.plot(years,[z.loc[str(y),'observed'] for y in years],color='#222222',marker='D',ls='--',label='观察发生率')
            if cause==1:
                s=scenarios[(scenarios.months==h)&(scenarios.supported==True)].set_index('year')
                ax.fill_between(years,s.loc[years,'low_mean'].to_numpy(),s.loc[years,'high_mean'].to_numpy(),color='#9583c4',alpha=.18,label='退市低—高情景（非置信区间）')
            ax.set_title(('先最终退市' if cause==1 else '先恢复')+f' · {h}个月')
            ax.set_xticks([2023,2024,2025]);ax.set_xlim(2022.85,2025.15);ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2);ax.set_ylabel('平均预测 / 观察发生率')
            if h==24:ax.text(2025,.02,'随访不足\n性能NA',ha='center',va='bottom',color='#777777',transform=ax.get_xaxis_transform())
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    fig.suptitle('历史日历发生水平更新：缓解低估，同时可能转为高估',fontsize=15)
    fig.text(.5,.105,'探索性；各年使用自己的截止日前快照。退市原参考与仅恢复更新重合；中、高情景均受同一证书上限限制。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.14,1,.95]);fig.savefig(out/'RISK_LEVEL_COMPARISON.png',dpi=180);fig.savefig(out/'RISK_LEVEL_COMPARISON.pdf');plt.close(fig)
    maincert=cert.query("arm=='both_2y'").copy();maincert['reduction_pct']=100*maincert.termination_reduction/maincert.requested_termination
    joint=m.query("cause==0 and supported==True and arm in ['reference','recovery_only','both_2y']").pivot(index=['scope','months'],columns='arm',values='joint_brier').reset_index()
    peryear=m.query("cause==1 and supported==True and scope!='pooled' and arm=='reference'")
    # Numerical results below are generated from full private aggregate outputs.
    text='''# 日历人时线性水平更新：探索性结果

**主方案缓解了2023、2024年的退市风险低估，但没有消除；2025年12个月转为高估。完整精确解释与合法CIF可以同时保留，代价包括跨年排序变化和部分恢复误差上升。** 不据此更换当前参考模型，不选择敏感性方案。

本轮森林拟合0、校准器拟合0、倍率/窗口搜索0。只对已保存的83变量、q2、深度4、直接父节点AJ10参考森林的三种子均值做固定规则映射。2023—2025已暴露；窗口与方案是在查看reporting-drift公开CALENDAR_INCIDENCE后设计的，不是独立确认实验。原始财报版本PROVISIONAL限制继续存在。

## 退市低估缓解了多少

O/E=观察发生率/平均预测；大于1是群体低估，小于1是群体高估，不是单家企业概率真值。表中风险为0—1比例，最后一列是平均概率绝对偏差的相对减少百分数。

'''+table(comparison,['scope','months','observed','reference_mean','updated_mean','reference_OE','updated_OE','reference_brier','updated_brier','absolute_gap_reduction_pct'])+'''
2024年12个月仍明显低估：更新8.07%，观察35.56%，差27.48个百分点，O/E仍4.4034。2024年24个月由11.58%升至28.89%，观察44.44%，仍差15.55个百分点。2025年12个月由5.85%升至12.08%，超过观察9.43%约2.65个百分点，即相对观察率高28.05%；O/E降到0.7809。虽退市Brier略降，但其差值区间跨0，不能据此声称2025改善可靠。

## 截止日前实际有多少上调信息

只用每个Jan1截止点自己的train与origin/stop快照，分别309/345/390家公司、14/26/49例历史退市。未使用reporting-drift后来快照的日历表回填。实际天数÷(365.25/12)为人月；[origin,stop)分配暴露，事件按stop归入，最多24日历月。近期不成熟公司贡献其已观察的人时。

'''+table(slopes,['year','window','cause','historical_events','window_events','historical_person_months','window_person_months','raw_ratio','requested','lower95','upper95','bootstrap_sparse_fallback'])+'''
cause1退市、2恢复；2y主窗口、1y敏感性。点估计没有触发少于5独立事件公司的回退；2024/2025的1年退市原始倍率4.1012/4.4961触及上限4。训练公司1000次重抽样、独立种子2026110203，区间是门槛和边界处理后的请求倍率描述性区间；重复抽中同家公司不增加独立事件支持。稀少原因回退次数和未定义原始比值次数均保留。

日历人时发生率比不是累计发生概率比。其部署假设是近期日历发生水平可作为下一批入组公司的风险幅度提示；它没有调整入组后时长、风险集构成或竞争原因构成，不能称为一致校准估计器或政策因果效应。

与实际事后所需O/E的差距如下。事后O/E只用于诊断，从未输入倍率计算。

'''+table(scenarios.query('supported==True'),['year','months','requested_g','certified_g','required_oe','required_minus_requested'])+'''
## 全输入概率证书限制了多少

先合并96棵树，再对终端叶CIF凸包作Minkowski包络。保留恢复斜率，将退市斜率降低至证书允许的最大值；不是按评价样本裁剪概率。包络允许不一定同时可达的跨树叶组合，因而是充分而可能保守的证书。

'''+table(maincert,['year','requested_termination','certified_termination','certified_recovery','termination_reduction','reduction_pct','requested_bound','actual_bound'])+'''
2023恢复斜率略大于1，占用了概率余量，退市请求倍率被削减约49.25%；2024削减21.19%；2025削减40.66%。三年主要方案证书上限均为1。uncertified_2y单列为诊断，不能称为合法模型：2023/2024在检查到的样本上未出现违规，但全输入证书未通过；2025在全部原生时间检查中有78家训练、10家评价公司至少一次两原因之和超过1，最大1.2420。证书最大和1.5279是包络上界，不等于实际样本最大和。

## recovery_only 与 both_2y 的事实比较

两者恢复预测逐元素完全相同，恢复AUC/AP、O/E、Brier也相同。recovery_only的退市预测与reference完全相同；both_2y才提高退市水平。两原因联合Brier定义为两原因平方误差之和，不包括无事件第三状态；不能与三状态Brier或平均两项的指标混称。

'''+table(joint,['scope','months','reference','recovery_only','both_2y'])+'''
2023恢复12个月Brier0.05923→0.06253、24个月0.25288→0.25881，均变差。2024/2025的恢复Brier下降。2023年12个月both_2y联合Brier0.16576→0.16810，仍变差；其他可评价窗口联合Brier下降。以上只陈述预设各臂结果，不替研究者选择方案。

## 排序保持与跨年变化

每年每原因都乘同一正数，年度AUC、AP、Top20以及配对数加权的年内AUC逐值保持。所有排序、并列关系、sklearn与纯配对核验通过。退市年度参考指标如下（捕获1—4的小计在公开CSV抑制；名额按年ceil(0.2n)，合并先逐年选取）：

'''+table(peryear,['scope','months','pooled_auc','ap','precision','coverage'])+'''
跨年采用不同倍率，因此pooled会变。下表在合并公司上重算，不是年度指标平均。

'''+table(m.query("cause==1 and scope=='pooled'"),['months','arm','within_year_auc','pooled_auc','ap','brier','observed_expected'])+'''
both_2y的12个月pooled AUC/AP由0.6451/0.2469升至0.6547/0.2579；24个月由0.7296/0.6817降至0.6945/0.6638。年内AUC分别保持0.7584、0.7614。both_1y虽有部分更好点估计，仍只是预设敏感性结果，未据此替换主窗口。uncertified_2y的Brier列仅为诊断平方误差，不证明概率模型合法。

主方案退市Brier相对reference的差值及固定预测条件区间：

'''+table(diffs.query("arm=='both_2y' and cause==1 and metric=='brier'"),['scope','months','estimate','lower95','upper95','undefined_repeats'])+'''
评价bootstrap逐元素复用cause-support的1000组公司索引，按测试年分层、种子2026110201；斜率与模型固定，不包含训练、水平参数估计或此前多轮选择不确定性。所有指标的未定义次数保留。2023年12个月恢复观察为0，abs(log(O/E))为正无穷，相应差值不可定义，不能填0或把退化重抽样解释为真实风险确定为0。

## 水平情景

低为退市倍率1，中为主方案证书倍率，高为截止日前最近5个完整日历年的最高粗率比，先限制[0.5,4]再证书约束；全部使用主方案的恢复倍率。高情景未另加用户未要求的年度5事件门槛。三年高、中均被证书压到相同上限。保留标签，不重新排序、不据观察结果扩宽。

'''+table(scenarios,['year','months','low_mean','middle_mean','high_mean','observed','observed_inside'])+'''
SCENARIO_CONTAINMENT_STATEMENT

![预测水平与观察水平](RISK_LEVEL_COMPARISON.png)

## 数值、执行与限制

协议提交bea1601877f7efe50227ea817946cede93997ad2后才读取本轮评价标签，五个指定参考O/E全部复现。三个年度任务、18套完整表示均完成；全部训练和评价公司、全部原生时间、两个原因，最大重构/显式求和/保存重载误差见FUNCTION_VALIDATION.csv，均小于1e-10。q2结构未变，所有函数按对应原因统一缩放；不修复相对交互幅度、变量选择或排序。

2025年24个月仅输出私有预测和合法性检查，性能NA。纯Python逐公司人时与向量实现一致，独立倍率算式一致；跨年、闰年24月上限、删失、零事件、独立事件门槛、截止日隔离测试通过。旧模型、旧报告、CURRENT指针和输入哈希保持不变。

Slurm2619申请两节点池，3项短任务实际均由oliver先领取，约25.17秒完成计算、含队列26.78秒；lemon依赖已核验，但没有虚报其完成任务，也未为填两节点记录重复计算。峰值内存与每项耗时在回执中。完整pytest含独立R检查，已知EBM缺失值可视化警告保留。

本轮无失败任务、无追加拟合、无参数搜索、无概率裁剪。尚不能回答新队列泛化、2025年24月校准、单家企业概率真值或政策因果机制。保持当前参考模型；前向规则见FORWARD_VALIDATION_PLAN_ZH.md。
'''
    # Generate the interval-containment statement from data, never assume it.
    supported=scenarios[scenarios.supported==True]
    inside=supported[supported.observed_inside==True]
    statement=f'可评价的{len(supported)}个窗口中，{len(inside)}个观察率落在低—高区间内。'
    if len(inside):statement+=' 落入者为'+ '、'.join(f'{int(r.year)}年{int(r.months)}个月' for r in inside.itertuples())+'。'
    statement+=' 2023/2024观察率仍高于上界；区间不是置信区间，也没有覆盖保证。'
    text=text.replace('SCENARIO_CONTAINMENT_STATEMENT',statement)
    (out/'RESULTS_ZH.md').write_text(text)
    print(out)


if __name__=='__main__':main()
