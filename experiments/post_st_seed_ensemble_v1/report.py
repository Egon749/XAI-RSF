"""Publish aggregate-only fixed-combination results."""
import datetime,platform,shutil,subprocess
import pandas as pd
import scipy,sklearn
from experiments.post_st_tree_count_v1.report import table
from .common import *
def main():
    r=run_path();pub=REPO/'experiments'/NAME;v=load(r/'reports/validation.json');assert v['status']=='PASS'
    verdict=(pub/'INTERPRETATION_ZH.md').read_text()
    for f in(r/'reports').iterdir():assert not(pub/f.name).exists();shutil.copy2(f,pub/f.name)
    d=pd.read_csv(pub/'ENSEMBLE_METRICS.csv');d['scope']=d.scope.astype(str)
    target=d[d.cause==1].copy();target['capture']=target.apply(lambda z:f'{int(z.captured)}/{int(z.target_events)}'if pd.notna(z.captured)else'NA',axis=1)
    target['arm']=target.arm.replace({'reference96':'原96树','requested7_224':'七种子224树','requested8_256':'八种子256树'})
    sections=[]
    for y in [*YEARS,'pooled']:
        f=target[target.scope==str(y)].sort_values(['months','trees']);f.to_csv(pub/f'TERMINATION_{y}.csv',index=False)
        cols={'months':'月数','arm':'组合','auc':'AUC','ap':'AP','brier':'Brier','capture':'Top20捕获','risk_mean':'平均风险','observed':'观察率'}
        sections.append(f'## {y}\n\n'+table(f,cols))
    (pub/'RESULTS_ZH.md').write_text(f'''# 固定种子组合预测：224树与256树

{verdict}

## 定义与执行

每家企业、同一时间与原因，分别对指定七种子1009/503/17/202/101/419/300的CIF求平均，及加入42后的八种子CIF求平均。每个成员32棵树，等价于224/256棵树等权平均；原42/101/202的96树组合单独保留。不是平均AUC，也没有按年份选不同种子、拟合权重或重新收缩。模型仍为74输入、q≤2、深度4、退市支持保护、父节点AJ收缩10，训练随访上限24个日历月。

实际新森林拟合0、校准拟合0；六个合并森林及完整函数表示由已保存成员导出。各年输入与1月1日历史信息边界不变。使用各公司日历终点计算12/24月风险，Top20名额2023/2024/2025分别8/9/11。恢复原因完整结果在ENSEMBLE_METRICS.csv；2023年12月恢复无事件时AUC/AP为NA。2025年24个月随访不足，性能NA。

{chr(10).join(sections)}

## 验证与解释

全部训练/评价样本、所有原生时间、两个原因核验合并预测等于成员预测均值，最大误差{v['max_mean_error']:.3e}。完整CIF重构最大误差{v['max_reconstruction']:.3e}，独立逐项求和误差{v['max_explicit_sum']:.3e}，保存重载、q≤2、非负、单调、两原因和≤1均通过，无裁剪。FUNCTION_VALIDATION.csv含主效应/交互/单元复杂度、编译耗时等；合并后依然精确，但函数数量可能增加，不能据此声称解释更简单。

原96树各年度/合并指标逐项复现。合并AUC/AP/Brier从合并预测重新计算，Top20先按年度选定后合计。PAIRED_DIFFERENCES_95.csv保留1000次原按年分层公司重抽样、seed2026110201的条件差值区间及不可估计次数，固定模型预测，不包含训练和种子选择不确定性。

指定名单来自已暴露的种子年度排名；本次仅为探索性组合复核。树数同时从96增加到224/256，不能把差异孤立归因于种子，也不能据其宣称新的独立验证或最优模型。风险排序改善不等于绝对概率校准改善。继承原数据时钟与变量选择限制，未修订输入。当前模型和历史报告全部保留，不合并分支或自动替换模型。
''')
    log=r/'audit/pytest_after_fix.log';txt=log.read_text();assert '283 passed'in txt and'skipped'not in txt;shutil.copy2(log,pub/'pytest_final.log')
    write(pub/'execution_receipt.json',dict(status='COMPLETED_EXPLORATORY_SAVED_ENSEMBLES',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE_COMMIT,
        code_commit=load(r/'audit/code_snapshot.json')['commit'],report_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),branch='research/xai-rsf-seed-ensemble-v1',
        private_run=str(r),arms=ARMS,forest_fits=0,calibration_fits=0,new_trees=0,derived_ensembles=6,reference96_preserved=True,
        gate=load(r/'audit/gate.json'),validation=v,input_hashes_sha256=sha(r/'audit/input_hashes.json'),source_hashes_sha256=sha(r/'audit/source_hashes.json'),
        tests='283passed after metadata fix;independentR included,no skips;inheritedEBM warning',failures=load(r/'audit/analysis_failure.json'),environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        public_output_hashes={f.name:sha(f)for f in pub.iterdir()if f.is_file()and f.name!='execution_receipt.json'}))
if __name__=='__main__':main()
