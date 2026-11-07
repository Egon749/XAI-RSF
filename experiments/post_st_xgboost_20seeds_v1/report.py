import pandas as pd
from experiments.post_st_c061_xgboost_v1.report import md_table
from .common import *

def main():
    m=pd.read_csv(HERE/'METRICS_ALL.csv');s=pd.read_csv(HERE/'SEED_DISTRIBUTION.csv')
    ensemble=m[(m.model=='XGBoost_random20')&(m.scope!='pooled')]
    single=m[m.model.isin([f'XGBoost/{seed}' for seed in SEEDS])].copy()
    single['seed']=single.model.str.split('/').str[-1].astype(int)
    annual=single[single.scope!='pooled'].copy();annual['task']=annual.scope+'/'+annual.months.astype(str)
    for metric in ['auc','ap','ipcw_brier','top20_captured']:
        wide=annual.pivot(index='seed',columns='task',values=metric).reindex(SEEDS)
        wide.to_csv(HERE/f'SEED_{metric.upper()}_WIDE.csv')
    aucwide=pd.read_csv(HERE/'SEED_AUC_WIDE.csv');aucwide['seed']=aucwide.seed.astype(str)
    aggregate=m[(m.scope=='pooled')&m.model.isin(['C061','XGBoost_previous3','XGBoost_random20'])]
    distribution=s[(s.scope!='pooled')&s.metric.isin(['auc','ap','top20_captured'])]
    freeze=read(run()/'snapshot.json')['commit']
    text=f'''# 随机20种子XGBoost重训结果

已完成全部120次最终XGBoost拟合，没有重新调参或改动C061。20种子平均预测的12个月AUC为2023年0.8359、2024年0.8901、2025年0.6833，Top20捕获3/4、7/16、2/5。24个月2023/2024年AUC为0.6806/0.6780，捕获4/12、6/20。相较上一轮42/101/202，2025年12个月排序改善，但其他年度/时限未一致改善；风险低估依然明显。没有根据本轮结果选种子、改参数或替换参考模型。

## 冻结和执行

基准提交1f1d3041ec4778a1844e8157c972981eb1ee2a86，拟合前协议及任务代码提交{freeze}。用Python独立随机发生器Random(2026110701)，从1至2147483646均匀无放回抽20个种子，只抽一次，保留抽取顺序，各年/时限共用。种子完整名单：{list(SEEDS)}。数值大小没有模型优劣含义。

原始历史参数选择保留：2023年12/24月深度1、192轮，lambda分别1/10；2024年12/24月深度1、32/96轮、lambda10；2025年12/24月深度2、192/32轮、lambda1/10。学习率0.05、行列抽样0.8、74项原生NaN输入、IPCW、标签和其他XGBoost设置不变。原先历史前向调参流程本次不重复，因此这是条件于既定参数和固定训练样本的算法随机性分析，不是完整筛选或训练公司重抽样。

训练数据为各自2023/2024/2025年1月1日前快照，309/345/390家公司；评价36/45/53家公司，来源文件逐字节哈希对齐。训练任务不接收评价结局。早恢复按既有定义为非退市，未知早删失不强行当负例。2025年24月仅保留预测/结构检查，全部性能为NA。

Slurm2703的24个互不重叠任务全部成功，每任务5个种子。oliver和lemon各12任务、60次拟合，端到端10.24秒。记录的24任务秒数不得与墙钟时间混淆。全120模型保存、重载后在全部训练和评价行的预测逐值相同；无额外真实森林或校准器拟合。原C061及两组3种子XGBoost均保留。

## 20种子组合预测

先平均20个模型对同一家公司的分数，再计算指标；不是平均20个AUC。Top20按年ceil(0.2n)，各年名额8/9/11。

{md_table(ensemble[['scope','months','auc','ap','ipcw_brier','top20_captured','target_events','observed_expected']])}

与3种子组合相比，本轮同时改变成员和集成数量，不能把组合差异完全归因于某个种子的优劣。所有20个种子均保留，无按年份挑选最优模型。

## 每个随机种子的AUC

下表按预先冻结的抽取顺序排列，没有按测试表现选取。AP、Brier、捕获的同款宽表及全部长表随附，2025/24月为空值。

{md_table(aucwide)}

## 单种子分布

下表为固定测试样本下20个随机模型的分布，中位数及5%–95%分位不是置信区间。样本数没有增加到20个独立队列。所有支持时限20个值均可估计；2025/24月20个均不支持，未填零。

{md_table(distribution[['scope','months','metric','defined','undefined','min','p05','median','p95','max']])}

2023年12月AUC范围0.7656–0.8750；2024年0.8351–0.8955；2025年0.5958–0.7083。2024年12月所有种子都捕获7/16，2023年24月都捕获4/12。这提示部分排名波动不会改变固定核查名额中的捕获数量。仅20次随机重复不能穷尽随机性，更不能将最大值当作可信泛化性能。

## 既有参照与合并评价

previous3为42/101/202；old3为17/155/8679；C061仍是保存的17/155/8679 XAI-RSF。两个旧组合及6个旧种子的原指标均独立复现。PRIOR_DISTRIBUTION_POSITIONS.csv给出这些参照在本次单种子指标分布中的升序中秩百分位；组合预测器和单模型复杂度不同，位置只作描述。

{md_table(aggregate[['months','model','auc','within_year_auc','ap','ipcw_brier','top20_captured','target_events']])}

12个月新组合的年内配对加权AUC0.8221与C061的0.8185接近，直接合并AUC0.8059高于C061的0.6789，两种口径不能混用。24个月新组合年内AUC0.6789低于C061的0.7792；Top20总捕获10/32，C061为14/32。

采用原1000次按测试年分层、公司级配对抽样索引，analysis seed2026110201。新组合减上一组3种子的12月年内AUC差0.0042，95%区间[-0.0098,0.0223]；24月差-0.0102，[-0.0318,0.0065]，均包括0。新组合减C061的24月年内AUC差-0.1003，[-0.2028,-0.0092]；12月Brier差+0.0097，[0.0028,0.0184]，方向不利于XGBoost。所有预定指标和未定义次数都保留，不选择性只讲有利区间。这些区间条件于固定预测，不包括训练数据、调参和既往种子选择不确定性。

## 概率、解释与限制

12月O/E为12.61、11.74、2.69，24月为7.51、3.79，仍为低估；随机种子扩展并不校准绝对水平。C061的完整二阶CIF分解未改变，但XGBoost没有该精确二阶解释性质。

独立的12月/24月分类器不保证随时间单调：新组合2023/2024/2025仍分别有1/4/4家的12月分数高于24月，最大差约0.0395/0.0107/0.2533。未裁剪或修正；2025年的该检查不使用24月绩效结论。XGBoost不同时预测恢复CIF，不能称为完整竞争风险曲线。

2023–2025结果已经反复暴露，C061存在既往组合选择的乐观偏差；本次只作探索性敏感性分析。沿用旧输入的PIT/历史财报版本限制，不重新宣称通过独立审计。没有更换主模型或寻找新参数。

## 失败记录和验收

第一次测试收集因新目录缺少包初始化文件而失败，补齐后完整测试通过，初始失败日志保留。最初lemon交互环境探测未设置项目的依赖目录，因此未找到XGBoost；使用实际启动器相同的PYTHONPATH后，版本3.4.1及NFS输入验证通过。这两项均发生在正式训练之前，没有删除训练失败或重复有利结果。正式120拟合与24任务失败均为0。

完整pytest含独立R检查、全模型重载、独立sklearn AUC/AP、直接Top20、原预测复现、输入/输出哈希和公开隐私扫描见日志/回执。企业数据、标识、模型和逐公司预测只留新私有目录。公共汇总保存全部20种子，不覆盖任何旧结果，分支不合并。
'''
    (HERE/'RESULTS_ZH.md').write_text(text)

if __name__=='__main__':main()
