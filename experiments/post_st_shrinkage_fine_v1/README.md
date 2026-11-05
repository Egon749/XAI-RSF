# C039父节点收缩强度0—20，步长1

这是查看粗网格结果后、由研究者指定的有限细网格探索。当前参考值10和全部旧模型保留。比较21个整数；复用0/1/2/5/10/20六个强度在2020—2025的36组输出，只新增其余15个强度的90组派生输出。直接读取已保存的18个raw森林，新增森林和校准器拟合均为0。

固定74输入、种子17/503/22各32树、q2、深度4、退市支持保护、共享分位切点和均衡25变量覆盖。仅变动直接父节点AJ权重 m/(m+lambda) 的lambda；同一叶节点两个原因全部时间使用同一权重，m为独立袋内公司数。不能重复收缩已收缩森林。

```bash
python -m experiments.post_st_shrinkage_fine_v1.prepare
# Commit frozen protocol/code, then stage immutable inputs.
python -m experiments.post_st_shrinkage_fine_v1.schedule
hetero-run PRIVATE_RUN/derive.jsonl
python -m experiments.post_st_shrinkage_fine_v1.analyze history
# Commit historical selection before outer evaluation.
python -m experiments.post_st_shrinkage_fine_v1.analyze outer
python -m experiments.post_st_shrinkage_fine_v1.report
```

重新复现须为 run_location.json 指定新唯一私有目录；排他创建以避免覆盖。原数据、公司预测、raw/派生森林和完整函数表均留私有目录。复用工件的位置记录在私有清单，公开仅汇总和哈希。

主指标为12个月退市年内配对加权AUC，同时报告AP、IPCW Brier、Top20、O/E；24个月及恢复为次要。历史验证仅用各截止日前可得且时限成熟的原训练快照，事件数只有4/4/8；外层2023—2025均已暴露，2025的24个月性能NA。完整记录单种子和先平均预测后计算的96树指标、1000原有配对公司抽样区间、函数复杂度/稳定性/数值验证。不据本轮结果自动更换主模型。
