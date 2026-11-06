"""Aggregate-only report and figures; never selects or refits a model."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

D=Path(__file__).resolve().parent
ARMS=['reference','temporal_affine','calendar_2y','calendar_damped','age_scale','age_mix25','age_mix50']
NAMES=['原模型','历史样本外仿射','近期日历发生率','温和日历更新','随访阶段水平比','近期曲线混合25%','近期曲线混合50%']

def table(frame,cols):
    s=['| '+' | '.join(cols.values())+' |','|'+'|'.join(['---']*len(cols))+'|']
    for _,r in frame.iterrows():
        cells=[]
        for c in cols:
            v=r[c]
            if pd.isna(v):v='NA'
            elif isinstance(v,(float,np.floating)):v=f'{v:.4f}'
            cells.append(str(v))
        s.append('| '+' | '.join(cells)+' |')
    return '\n'.join(s)

def main():
    f=pd.read_csv(D/'METRICS_ALL.csv',dtype={'scope':str})
    v=pd.read_csv(D/'FUNCTION_VALIDATION.csv')
    pp=pd.read_csv(D/'PAIRED_DIFFERENCES.csv',dtype={'scope':str})
    tiers=pd.read_csv(D/'RISK_GROUP_CALIBRATION.csv',dtype={'scope':str})
    joint=pd.read_csv(D/'JOINT_BRIER.csv',dtype={'scope':str})
    (D/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':42})
    colors=['#333b4a','#9164b0','#007f78','#5b9ea0','#d08b19','#70a0cd','#ca8b9b']
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
    for ax,h,ys in zip(axes,[12,24],[['2023','2024','2025'],['2023','2024']]):
        x=np.arange(len(ys))
        for a,name,color in zip(ARMS,NAMES,colors):
            z=f[(f.months==h)&(f.cause==1)&(f.arm==a)].set_index('scope').loc[ys]
            ax.plot(x,z.mean_predicted,'o-',label=name,color=color,linewidth=1.7,markersize=4)
        z=f[(f.months==h)&(f.cause==1)&(f.arm=='reference')].set_index('scope').loc[ys]
        ax.plot(x,z.observed,'D--',color='#bd342e',label='观察发生率',linewidth=2,markersize=6)
        ax.set(xticks=x,xticklabels=ys,title=f'{h}个月退市：年度平均预测与观察',xlabel='入组年')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('退市累积发生概率 / 观察发生率')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.suptitle('固定主模型的概率水平更新：能缓解低估，仍有跨年错配',fontsize=14)
    fig.tight_layout(rect=[0,.14,1,.94])
    for ext in ['png','pdf']:fig.savefig(D/f'figures/level_comparison.{ext}',dpi=180)
    plt.close(fig)
    annual=f[(f.scope!='pooled')&(f.cause==1)&~((f.scope=='2025')&(f.months==24))]
    pooled=f[(f.scope=='pooled')&(f.cause==1)]
    recovery=f[(f.scope!='pooled')&(f.cause==2)&~((f.scope=='2025')&(f.months==24))]
    ci=pp[(pp.cause==1)&(pp.arm=='calendar_2y')&(pp.metric=='ipcw_brier')]
    cols={'scope':'年份','months':'月','arm':'方案','mean_predicted':'平均预测','observed':'观察率','observed_expected':'O/E','ipcw_brier':'Brier'}
    text="""# C039退市概率水平探索结果

本轮确实缓解了概率低估，同时保住了年度排序与完整q2解释，但没有找到跨年度均充分校准的通用修复。近期日历发生率更新在2024年效果最明显：12个月平均预测3.62%→9.13%（观察35.56%），24个月11.96%→30.16%（观察44.44%）。同一规则使2025年12个月平均预测5.62%→11.71%，高于观察9.43%。2024年12个月仍严重低估。全部结果是已暴露队列上的探索性比较，原C039未替换，不能把本轮相对较好的方案称为已确认最优模型。

## 1. 实际执行与基线

基线是当前74输入C039，种子17/503/22各32树、q≤2、深度4、退市支持保护、共享切点、均衡25变量池、直接父节点AJ强度10。训练与评价输入、时钟、竞争事件和24日历月训练上限均沿用原冻结口径。2023/2024/2025训练公司分别309/345/390，训练退市14/26/49，恢复184/198/211；评价公司36/45/53。

先提交冻结协议，再计算任何新增评价结果。三年度×七臂=21组概率映射、完整函数导出与验证全部完成。新森林拟合0，历史样本外约束仿射拟合3，新增阈值/窗口/强度搜索0。九个主模型成员及原预测哈希核对通过，参考预测与原三种子均值完全一致。历史样本外校准预测来自同规格已保存森林，各截止日前有99/131/167家公司；少数退市事件导致三年退市斜率均按预设门槛固定1。没有把未知结局填0。

Slurm2700申请了oliver、lemon两节点，3个短年度任务实际均被oliver取走，耗时约26秒，完成并发布3/3。本轮不包含昂贵森林重训，不能声称两机分别完成了训练或充分占满所有CPU。

## 2. 固定方案及数学性质

所有方案形式为 F*_k(t,x)=b_k F_k(t,x)+a_k(t)，b_k>0，不随时间改变，a为非负单调公共曲线。完整函数精确变为 μ*=bμ+a、f*j=bfj、f*jl=bfjl。不重新估计树结构，不删非零函数，不裁剪预测。使用全输入终端包络证书限制两个原因概率和；这是一项充分条件，可能保守。

- reference：原模型。
- temporal_affine：使用各自截止日前已成熟历史队列的样本外预测，拟合既有约束仿射，20公司身份先验，原因事件不足10时固定斜率1。截距从365天起生效，所以12个月拟合向24个月的延伸也是预设探索，不能声称24个月专门校准。
- calendar_2y：最近两年日历事件/人时相对全历史的比率；每个原因少于5事件则1，参数范围0.5–4，经全输入证书约束。
- calendar_damped：上述请求倍率取平方根，再做相同合法性检查。
- age_scale：按入组后0–12/12–24个月分离历史与近期风险暴露，使用5个总伪事件的预定收缩；由早期累积发生率比得到常数倍率，稀少原因回退1。
- age_mix25/50：原预测与近期分阶段竞争风险曲线固定混合25%/50%，两原因共同凸组合。近期曲线使用月度阶梯网格，在365/366及730/731天取相同值。完整函数仍精确，但个体主效应和交互幅度分别缩小25%/50%。

窗口和方案基于此前已暴露的研究结果设计。参数计算只读各截止日自身训练快照，没有用2024或2025评价结局回填更早参数。未尝试逐年挑赢家、扩大倍率或放宽证书。

## 3. 逐年退市概率与误差

下表为概率小数。O/E为观察发生率除以平均预测，1理想，>1低估，<1高估。Brier越低越好。支持的评价终点均已知，IPCW权重为1，所以此处与普通二元Brier数值一致；竞争原因事件作为该原因的非事件，未把未知删失当正常。2025年24个月仅保存预测，性能NA。

"""
    text+=table(annual,cols)
    text+="""\n\ncalendar_2y使2024年24个月Brier由0.33663降至0.23408，改善0.10255；12个月由0.32432降至0.28246，改善0.04186。2025年平均概率相对观察高约24.2%，绝对高2.28个百分点；Brier仍由0.08370略降至0.08098，但其条件差值区间包含0。总体均值更近与个体概率误差更低不是同一个判断。

calendar_damped的2025年过估较轻（10.49% vs 9.43%），但2024年修复也更弱。temporal_affine保持退市贡献幅度，2025均值8.52%较接近观察，2024仅提高到5.09%，不足以跟上当年的35.56%。age_mix25在2025均值8.99%看起来接近，2024却几乎不能补足缺口，不能按单年挑选它。

## 4. 前20%企业与个体问题

2024年原模型前20%企业的24个月平均预测为17.11%，calendar_2y提高至43.13%；12个月为5.26%→13.26%。它们仍是同一组企业，捕获与排序完全不变。这直接对应“高风险企业的概率也只有十几”的疑问：数学上可以把概率水平提高，同时保住完整解释，但提高后的数值尚未充分校准。

新增风险分组表对1–4事件及非事件互补小格统一抑制，保留私人完整统计及公开平均预测。全年度既有汇总口径保留。不发布企业标识、逐公司预测或真实函数切点。

某一家企业后来发生退市，并不意味着其事前概率应为100%。是否低估应看可比企业组的预测与观察发生率。总体平均校准改善也不能证明每一家企业的概率已准确。

## 5. 排序、解释和跨年代价

全部年度、原因和支持时限的排序、并列关系、AUC、AP及Top20与原模型一致。退市年度AUC/AP：2023H12为0.8203/0.3791，2024H12为0.8750/0.8596，2025H12为0.7750/0.2703；2023H24为0.7778/0.7002，2024H24为0.7720/0.7770。不能说水平更新提高了年度识别能力。

不同年度采用不同映射，所以跨年度混合排序可能改变：
"""
    text+=table(pooled,{'months':'月','arm':'方案','auc':'pooled AUC','within_year_auc':'年内配对AUC','ap':'pooled AP','ipcw_brier':'Brier'})
    text+="""\n\n尤其age_mix25/50令12个月pooled AUC从0.6983降至约0.56，尽管每年AUC不变；近期曲线把2025整体抬得比2024更多，跨年风险水平排序恶化。calendar_2y的12个月pooled AUC/AP略增，24个月pooled AUC/AP下降，也必须保留。

三年主效应数58/59/55，二元函数134/123/120，本轮七臂均保持对应年度原函数结构。全训练与评价样本、所有原生时间与新增阶梯时间、两个原因，最大重构/逐项求和/重载误差为"""
    text+=f"{v[['reconstruction','component_sum','reload']].to_numpy().max():.3g}，小于1e-10；所有概率非负、随时间不减且两原因和≤1。"
    text+="""\n\n精确解释保留不等于解释的经济幅度原封不动。纯倍率会同比放大原贡献，公共截距进入基线；混合把原贡献缩小25%/50%，不能据此宣称交互恢复更好。这里没有拟合代理解释或赋予因果意义。

## 6. 恢复原因和联合误差

"""
    text+=table(recovery,cols)
    text+="\n\n两原因联合Brier定义为两个原因二元Brier之和，不包含第三项未发生事件状态的平方误差；不能与其他多分类Brier定义直接混用。\n\n"
    text+=table(joint[joint.scope=='pooled'],{'months':'月','arm':'方案','joint_brier':'联合Brier','difference':'减原模型','lower95':'95%下限','upper95':'95%上限'})
    text+="""\n\nage_mix25/50虽然在退市上部分改善，却恶化2023年24个月联合Brier（+0.01008/+0.02244），主要与恢复估计有关。calendar_2y在2023H12联合Brier也略增0.00019。没有一项设计能据单个退市指标宣称全面占优。

## 7. 不确定性、证书和局限

以下列出calendar_2y减原模型的退市Brier差值，负值改善。其他所有臂与指标均见PAIRED_DIFFERENCES.csv，未筛除不利结果。
"""
    text+=table(ci,{'scope':'范围','months':'月','difference':'差值','lower95':'95%下限','upper95':'95%上限'})
    text+="""\n\n沿用1000次公司抽样、按测试年分层、共同索引和seed2026110201。区间条件于已保存森林和已估计映射，不含训练或水平参数估计的不确定性，也未校正多方案比较。C039本身曾根据已暴露结果选择种子组合，本轮未消除该乐观性。

calendar_2y的退市请求上调受到全输入证书限制，2023/2024/2025分别削减约37.3%/20.4%/40.1%；age_scale的2025也受限制。证书通过证明数学合法性，不证明统计校准。即使取消证书，截止日前近期信号也不足以自动补足2024年12个月的全部缺口，本轮没有输出非法模型来制造更好结果。

年龄分层近期曲线的12个月退市水平在2023/2024/2025约2.92%/5.71%/19.09%，与对应已观察评价年11.11%/35.56%/9.43%不匹配。最近历史存在滞后，更新会在跃升年跟不上、回落年追高；这一时间模式与结果一致，但不能区分或证明政策的因果作用。原财务披露版本/PIT限制继续继承。低事件支持、稀少历史成熟结局以及仅三年评价都限制稳健结论。

## 8. 判断与执行记录

可以写：预先固定、仅依赖截止日前信息的概率水平模块能在保留年内排序和精确低阶函数表示的同时，部分缓解退市概率低估。可以写出2024年的实测改善和2025年的过估代价。

不能写：绝对概率已可靠校准、每个企业概率已准确、解释幅度恢复了真实机制、模块提高了年度AUC、某臂已经通过独立验证或应替换主模型。当前结论是部分缓解、跨年不稳，继续保留原C039。

执行中初次完整测试因未接入既有xgboost依赖目录而收集失败，修正运行环境后全部通过；首次汇总误将发布清单的目录条目当三列文件记录，已修复并重新核对全部文件哈希。均保留原日志，无森林重训、参数更改或失败臂删除。21组数值验证无失败。完整测试与版本、输入输出哈希见execution_receipt.json；源码与汇总独立分支同步，不合并、不替换当前指针。
"""
    (D/'RESULTS_ZH.md').write_text(text)
    print('report and figures saved')

if __name__=='__main__':main()
