"""Render the public narrative from aggregate tables."""
import pandas as pd
from .run import D


def fmt(value, digits=3):
    return 'NA' if pd.isna(value) else f'{value:.{digits}f}'


def main():
    summary = pd.read_csv(D / 'SIDE_BY_SIDE.csv', dtype={'scope': str}).set_index(['scope','months','model','metric'])
    detail = pd.read_csv(D / 'DISTRIBUTIONS.csv', dtype={'scope': str}).set_index(['scope','months','model','metric'])
    scopes = [(year, h) for h in (12, 24) for year in ('2023','2024','2025','pooled')]
    names = ['PROBoost', 'Stacking']
    rows = []
    def line(txt=''):
        rows.append(txt)
    def label(scope):
        return '合并' if scope == 'pooled' else scope
    def interval(row):
        return f"{fmt(row['median'])} [{fmt(row.p05)}, {fmt(row.p95)}]"
    line('# ESWA 退市预测对历史训练公司扰动的稳定性')
    line()
    line('**2023 年的两项 12 个月高分都随训练公司构成而变化，Chen Stacking 尤其明显。** 在 47 次可以拟合的重复中，PROBoost 有 11 次达到或超过原 AUC 0.9922，Chen 有 9 次达到或超过原 AUC 0.9297。两者均能在部分抽样下重现高排序；中位 AUC 分别为 0.9219 和 0.7031。该年度固定测试集只有 36 家公司、4 例退市。实验支持报告模型对训练样本敏感，尤其不能把一次接近完美的排序当作稳定水平。仅凭这 4 例无法判定原训练过拟合、泄漏，或把结果归因于某一政策变化。')
    line()
    line('两个节点接收了全部 600 个预设任务。585 个产生完整预测，6 个因独立事件公司不足无法开始所需交叉验证，9 个因内层 IPCW 权重不可估而中止。全部计入分母，没有补抽。使用同一年度同一组测试公司，未选择有利的重复，也未改动 C039。以下均为在已暴露队列上的探索性结果。')
    line()
    line('## 年度和合并测试排序')
    line()
    line('年度 AUC 按该年企业计算；合并行的 AUC 只比较同一年内的事件–对照对。AP 在合并行用所有公司得分重算。括号内为训练扰动的第 5 和 95 百分位。原始位置是原单次拟合 AUC 在有效重复中的经验百分位，同分计半。2025 年 24 个月仍可评分，性能留空。')
    line()
    line('|入组年|月|方法|有效/计划|原 AUC|原始位置|AUC 中位数 [P5, P95]|AUC 最小–最大|AP 中位数 [P5, P95]|')
    line('|---|---:|---|---:|---:|---:|---|---|---|')
    for sc,h in scopes:
        for m in names:
            a = summary.loc[sc,h,m,'test_within_year_auc']
            p = summary.loc[sc,h,m,'test_ap']
            line(f"|{label(sc)}|{h}|{m}|{int(a.finite)}/50|{fmt(a.original,4)}|{fmt(a.original_midpercentile,1)}|{interval(a)}|{fmt(a.minimum)}–{fmt(a.maximum)}|{interval(p)}|")
    line()
    line('跨年直接合并 AUC 会比较不同截止点模型给出的分数。下表将它与分年度选定名额的捕获数一起列出。12 个月有 25 例退市，24 个月有 32 例。捕获分位数可因插值为非整数，每次实际捕获均为整数。')
    line()
    line('|月|方法|原跨年 AUC|跨年 AUC 中位数 [P5, P95]|原 AP|原捕获数|捕获数中位数 [P5, P95]|')
    line('|---:|---|---:|---|---:|---:|---|')
    for h in (12,24):
        for m in names:
            a,p,c = [summary.loc['pooled',h,m,'test_'+f] for f in ('auc','ap','top20_captured')]
            line(f'|{h}|{m}|{fmt(a.original)}|{interval(a)}|{fmt(p.original)}|{fmt(c.original,0)}|{interval(c)}|')
    line()
    line('## 内部验证能否预告时间外表现')
    line()
    line('每次从源训练集抽取与原公司数相同的次数（2023、2024、2025 各 309、345、390 次）。同一家公司被重复抽中时保留训练权重，各层交叉验证把其所有副本放在同一侧。训练仅使用该年 1 月 1 日之前的快照。对完全未被该次抽中的历史公司评价最终完整模型，再与当年未来入组公司的测试表现配对。下表中的差为逐次相减再取中位数；历史队列的退市比例、删失、分布与未来队列不同，故差距不能直接当作过拟合量。')
    line()
    line('|年|月|方法|可配对 AUC 数/50|袋外年内 AUC 中位数|时间外年内 AUC 中位数|配对 AUC 差中位数 [P5, P95]|配对 AP 差中位数|')
    line('|---|---:|---|---:|---:|---:|---|---:|')
    for sc,h in scopes:
        for m in names:
            o,t,g,p = [detail.loc[sc,h,m,f] for f in ('oob_within_year_auc','test_within_year_auc','gap_within_year_auc','gap_ap')]
            line(f"|{label(sc)}|{h}|{m}|{int(g.finite)}/50|{fmt(o['median'])}|{fmt(t['median'])}|{interval(g)}|{fmt(p['median'])}|")
    line()
    line('2023 年 12 个月两种方法各有 47 次外层结果，但仅 45 次可计算袋外 AUC（另两次袋外没有退市）。PROBoost 当年袋外减测试 AUC 的配对中位数为 −0.134，Chen 为 +0.026。跨三年后，两者的年内 AUC 配对差中位数分别为 +0.172 和 +0.192。单一年度的异常高分、训练样本扰动和跨期泛化差距是三个不同的问题。')
    line()
    line('## 得分颗粒度与预警含义')
    line()
    line('2023 年 12 个月，PROBoost 原预测捕获 4/4，重抽后捕获中位数为 3/4，有 23/47 次捕获全部 4 家。Chen 原预测捕获 3/4，中位数为 2/4，有 7/47 次捕获全部 4 家。一个事件的名次跨过前 20% 边界就会使覆盖率改变 25 个百分点。')
    line()
    line('在该年 36 家企业中，PROBoost 的不同得分个数中位数为 9，得分极差中位数为 0.1446。Chen 的不同得分个数中位数为 34，但最高分减最低分的中位数仅约 0.000087，原单次也仅约 0.000088。许多分数可以彼此不同、排序也可以很高，同时分值几乎聚在一起。不能用 AUC 推断这些分数准确刻画了企业间的绝对退市概率差。逐次唯一值数按原数值精度计算，未先行四舍五入。')
    line()
    line('## 与 XAI-RSF 120 个组合并列')
    line()
    line('|入组年|月|PROBoost 年内 AUC 中位数 [P5, P95]|Chen 同指标|XAI-RSF 120 组同指标|C039|原固定 42/101/202|')
    line('|---|---:|---|---|---|---:|---:|')
    for sc,h in scopes:
        a,b,x = [summary.loc[sc,h,m,'test_within_year_auc'] for m in names+['XAI-RSF/120']]
        line(f'|{label(sc)}|{h}|{interval(a)}|{interval(b)}|{interval(x)}|{fmt(x.original)}|{fmt(x.fixed_anchor)}|')
    line()
    line('XAI-RSF 重用训练公司固定时 10 个种子生成的 120 个重叠三种子组合。C039 为在已暴露 12 个月结果上选出的组合，固定锚点单列。PROBoost 与 Chen 每次变动训练公司的组成、重复数并重跑原搜索和拟合。这两列分布的宽度回答不同问题，不能据此检验谁对同一种训练扰动更稳健。')
    line()
    line('## 失败、支持与复核')
    line()
    line('2023 年 12 个月，原历史训练集中仅 5 家已知退市。50 个抽样中的 3 个留下少于 2 家独立退市公司，两种方法各有 3 次无法启动所需分层交叉验证。Chen 在 2024 年 24 个月中另有 2 次、2025 年 24 个月中另有 7 次出现内层 IPCW 正性失败（所需删失生存概率为零），没有用有限常数替代。合并 12 个月两种方法各有 47 次完整预测；合并 24 个月分别有 50 和 48 次。2025 年 24 个月未参与任何性能汇总。分位数条件于对应指标可计算的重复，失败率也应一起报告。')
    line()
    line('逐次指标、袋外与外层差距、原始调参分数、得分唯一值数和极差见 [ALL_REPLICATES.csv](ALL_REPLICATES.csv)。所有指标的有效数、分位、范围、原始值与经验位置见 [DISTRIBUTIONS.csv](DISTRIBUTIONS.csv)。原始全样本拟合没有天然配对的袋外队列，原始袋外位置留作 NA。全部任务状态及原因见 [TRAINING_STATUS.csv](TRAINING_STATUS.csv)。')
    line()
    line('输入沿用原基准冻结的 74 项、结局和时间钟，原始披露版本和部分历史审计字段可得日的既有限制仍在。两种方法的完整单次基准均已复现，此处指由归档预测复算指标。训练公司、测试公司、训练快照及评分对齐均经核对，同一公司的副本不跨内部折。2023–2025 均为已暴露队列。可以在论文中报告训练公司变化后的中位数、分位范围、支持失败率和跨年差异；这些描述性分位数不提供新队列上的性能保证，也不构成更换主模型的依据。')
    (D / 'RESULTS_ZH.md').write_text('\n'.join(rows)+'\n')


if __name__ == '__main__':
    main()
