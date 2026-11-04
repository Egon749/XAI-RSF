"""Generate publication tables from the sealed aggregate outputs."""
from .common import *
import pandas as pd
RESULTS_COMMIT='c377096fd51db00f6ada45246bcd7eff1eb37542'
NAMES={'XAI/C039':'XAI-RSF C039','XAI/anchor':'XAI-RSF 原三种子','XAI/median120':'XAI-RSF 120 组中位数','AJ':'无协变量 AJ','FG':'惩罚 Fine–Gray','CSC':'惩罚原因别 Cox','KNN_AJ':'最近邻 AJ','RF/selected_seeds':'标准 RSF（指定种子）','RF/anchor_seeds':'标准 RSF（原种子）','unrestricted/selected_seeds':'不限 q（指定种子）','unrestricted/anchor_seeds':'不限 q（原种子）'}
def fmt(z):return 'NA' if pd.isna(z) else f'{z:.4f}'
def table(cols,rows):
    return '\n'.join(['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def main():
    m=pd.read_csv(D/'METRICS_ALL.csv',dtype={'scope':str});p=pd.read_csv(D/'PAIRED_DIFFERENCES.csv',dtype={'scope':str});c=pd.read_csv(D/'CALIBRATION.csv',dtype={'scope':str})
    text=['# 论文第 4.3、4.4 节：竞争风险模型比较与概率校准',f'**12 个月先退市的年内 AUC，C039 为 0.8377。** 相对无协变量 AJ、最近邻 AJ 和同种子不限 q 森林，点估计更高，条件 95% 差值区间不含 0；相对惩罚 Fine–Gray、原因别 Cox 和标准 RSF，点估计更高，但区间均包含 0。相对另一套种子的不限 q 森林，区间也包含 0。120 个固定组合的中位数为 0.7951，低于标准 RSF 的 0.8053。C039 的 12 个月平均退市预测为 4.04%，观察率为 18.66%（O/E 4.62）。论文可报告完整二阶解释下具有竞争力的年内排序，以及对简单基线的优势；现有证据不足以宣称普遍优于标准 RSF、概率校准良好或风险指数修正了低估。结果数据及代码提交 `{RESULTS_COMMIT}`。']
    text+=['## 4.3　相同公司与时间切分下的比较','2023、2024、2025 入组企业分别为 36、45、53 家。各年 1 月 1 日前的训练公司对应为 309、345、390 家。每家以入组公告后的下一交易日开盘前为原点，沿用两原因及删失定义，训练最长 24 个日历月。12 个月主分析有 25 家先退市，8 家先恢复；2023/2024 的 24 个月次分析有 32 家先退市、31 家先恢复。支持窗口的 IPCW 权重全部经核对为 1。2025 年 24 个月只核查预测与函数，性能为 NA。']
    for h in (12,24):
        z=m.query("scope=='pooled' and cause==1 and months==@h").set_index('model')
        text+=['### '+str(h)+' 个月先退市',table(['模型','年内 AUC','跨年合并 AUC','合并 AP','IPCW Brier','Top20 捕获','O/E'],[[v,*[fmt(z.at[k,i]) for i in ('within_year_auc','auc','ap','ipcw_brier')],f"{int(z.at[k,'top20_captured'])}/{25 if h==12 else 32}",fmt(z.at[k,'observed_expected'])] for k,v in NAMES.items()])]
    text+=['年内 AUC 仅比较同年配对，按各年事件数乘非事件数加权。合并 AUC、AP 直接用合并企业预测计算。Top20 先按年向上取整，12 个月共核查 28 家，24 个月共 17 家。AJ 在同年基本给出相同风险，其同分按冻结样本键排序，捕获数不代表个体区分能力。120 组中位数行按列取中位数，没有对应的单一森林或公司名单。','C039 与同种子标准 RSF 的 12 个月前 20% 分别捕获 13 和 10 例；标准 RSF 的跨年合并 AUC、AP 和 Brier 点估计均略优于 C039。Fine–Gray 的合并 AP 为 0.3294，也高于 C039 的 0.2883。年内排序、跨年排序和概率误差衡量的是不同性质。']
    text+=['### 相同公司的配对差值','差值均为左侧 XAI-RSF 参照减所列对比模型，括号为 1,000 次固定预测、按年分层公司重抽的 95% 百分位区间。指定种子是 17/503/22，原种子是 42/101/202。各随机模型也逐种子保存在 METRICS_ALL。']
    z=p.query("scope=='pooled' and cause==1 and metric=='within_year_auc'").set_index(['months','reference','comparator'])
    rr=[]
    for model in [n for n in NAMES if not n.startswith('XAI/')]:
        v=[NAMES[model]]
        for h,ref in [(12,'C039'),(12,'median120'),(24,'C039'),(24,'median120')]:
            k=z.loc[h,ref,model];v.append(f'{k.difference:+.4f}（{k.lower95:+.4f}, {k.upper95:+.4f}）')
        rr.append(v)
    text.append(table(['对比模型','12 月 C039','12 月中位数','24 月 C039','24 月中位数'],rr))
    text+=['C039 曾按已经查看的测试年 12 个月年内退市 AUC 从 120 组中选出。相同测试数据上，120 组该指标的 P05、中位数、P95 分别为 0.7740、0.7951、0.8246；原三种子为 0.7921。两种随机模型都用同样两套种子，但它们没有接受同样的 120 组选优；中位数对照揭示这种不对称，无法消除选优偏差。每次公司重抽都重新计算 120 组各自的指标再取中位数。区间不包含训练、调参或模型选择的不确定性，多个对照的区间不能作为独立的确认性检验。','### 次要结局：先恢复']
    rr=[]
    for name,label in NAMES.items():
        row=[label]
        for h in [12,24]:
            t=m.query("scope=='pooled' and cause==2 and months==@h and model==@name").iloc[0]
            row.extend([fmt(t.within_year_auc),fmt(t.ap),fmt(t.ipcw_brier)])
        rr.append(row)
    text+=[table(['模型','12 月年内 AUC','12 月 AP','12 月 Brier','24 月年内 AUC','24 月 AP','24 月 Brier'],rr),'C039 的恢复排序没有重现退市排序的优势。12 个月合并年内恢复 AUC 为 0.5571，惩罚原因别 Cox 为 约 0.68；2023 年 12 个月未观察到先恢复，该年的恢复 AUC、AP 不可估计，也不贡献年内恢复比较对。具体点估计见表。','### 调参与函数复杂度','对每个外层截止年 Y，只取 Y−3 和 Y−2 入组队列作内部验证，并将相应拟合数据重新截到该内部年份 1 月 1 日前。最早内部拟合只有 2 例退市；2022 验证队列无 12 个月退市。这限制了调参精度。三期均选中岭罚 1、标准 RSF 最小节点参数 20、最近邻 k=20。罚强度按各原因每个事件的平均部分似然定义。AJ 和同后端不限 q 臂使用固定配置。后者只改 path_budget，其余训练和直接父节点收缩与主模型一致。','C039 的三个年度模型分别有 58/59/55 个非零主效应、134/123/120 个非零二元项，完整空间单元 2790/2638/2577，无三、四阶项。同种子不限 q 的对应空间单元为 80632/84682/92975，三个年度均实际出现三、四阶项。标准 RSF 原生树预测可复现，但本研究没有给它导出完整的低阶 CIF 表；不能把未导出记为 0 个函数。回归系数只在线性预测子尺度可加，邻居 AJ 按企业邻域估计。详细函数数和耗时见 COMPLEXITY、COMPUTE_COST。','两个 Fine–Gray 边际分别拟合，单原因曲线合法。在全部可评价时限没有发现双原因 CIF 和超过 1；2025 年 24 个月的数值核验发现 2 家超过 1，最大 1.0422。这一诊断保留原预测，不用于该时限的性能评分。']
    text+=['## 4.4　概率水平与风险分档','O/E 是观察发生率除以平均预测。校准差取预测减观察，单位为百分点。O/E 大于 1 表示平均预测偏低。总体与逐年区间见 CALIBRATION。']
    rr=[]
    for r0 in c.itertuples():
        if not r0.supported:continue
        rr.append([r0.scope,r0.months,'退市' if r0.cause==1 else '恢复',f'{r0.mean_predicted*100:.2f}%',f'{r0.observed*100:.2f}%',f'{r0.calibration_gap*100:+.2f}',fmt(r0.observed_expected)])
    text+=[table(['入组年','月','原因','平均预测','观察率','校准差（百分点）','O/E'],rr),'12 个月退市合并 O/E 的条件 95% 区间为 3.18–6.19，平均低估 14.62 个百分点。2024 低估最明显，2025 仍低估。24 个月在有随访支持的两年中平均预测为 9.78%，观察为 39.51%。恢复则在合并层面高估。','### 历史袋外阈值','同年同原因使用共同分母：12 月取训练边际乘积基准 CIF 的第 365 天，24 月取第 730 天；分子使用企业自己的日历终点。指数等于 100 乘原 CIF 除以这个分母。训练企业只使用未抽中它的树，门槛为至少 10 棵树、至少 30 家训练公司。三个快照的 309、345、390 家训练企业全部符合，实际每家 19–49 棵袋外树。用训练 OOB 指数的中位数和第 80 百分位分档，边界并列归入上档。','12 个月退市高档在 2023/2024/2025 分别有 8/36、15/45、7/53 家（22.2%、33.3%、13.2%）。合并高档 16/30 退市（53.3%），其余 9/104（8.7%），发生率差 +44.68 个百分点，条件 95% 区间 +26.28 至 +61.96。各年风险分布不同，固定历史阈值与当年 Top20 名额是两种预警规则。逐年事件小格及可反推出它的互补格已抑制。','所有训练和评价企业、全部原生时点及两个原因均完成逐项相加和保存重载。指数重构最大绝对误差 1.60×10⁻¹² 以下，反变换回 CIF 的误差小于 1.7×10⁻¹⁵；支持窗口的年度排序、并列、AUC、AP、Top20 与原模型相同。指数不是事件概率，不适合直接跨年比较。没有在测试集拟合校准器，尺度转换也没有缩小原概率误差。','## 执行与局限','Slurm 2680 在 oliver 和 lemon 分别完成 17 和 16 项独立作业，33/33 成功并回传。标准 RSF 新拟合 36 个成员森林（其中 18 个内部验证），同后端不限 q 拟合 18 个正式森林。C039 新增拟合为 0。对 9 个保存成员的森林、预测及函数逐项核对原哈希；430 个可评价的公司—时限—原因预测和独立归档完全相等。训练期信息、拟合迭代、未定义重抽次数及数值诊断均已保存。','调度后曾有两次报告程序错误，一次将不可估 AUC 序列化为非标准 JSON，另一次把发布清单的目录项当成文件项。修正后由相同的已保存预测重放评价并核对全部发布哈希，没有重新训练。冻结了既有 74 项输入及其已知历史版本限制。2023–2025 队列和选种结果已经暴露，本研究用于回顾性论文分析。']
    (D/'RESULTS_ZH.md').write_text('\n\n'.join(text)+'\n')
if __name__=='__main__':main()
