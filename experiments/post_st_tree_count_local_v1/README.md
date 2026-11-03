# 80—120棵范围的有限局部搜索

保留三种子同树数、等权平均，检查合并81、84……120棵（每种子27—40），共14档。80无法由三个相同整数树数构成，因此不改变种子权重去凑80。固定74变量、退市支持保护、q2、深度4、共享切点、均衡25变量、直接父节点AJ收缩10、24个月训练随访。

全部候选由已保存256树/种子森林的原编号前缀构造，零新增拟合。复用原96树的12份单种子/合并导出，新增156项raw/收缩联合导出任务，包含一个120树资源/数值试运行。完整336份表示均核验全部训练/评价样本、全部原生时间、两个原因的精确重构与概率合法性。

本轮在查看上一轮树数结果后提出，属于已暴露2023—2025样本上的局部优化。预先固定按12个月年内退市AUC排序，数值并列时依次看逐年Top20合计、pooledAP、较低Brier、较少树数；这仅是探索性点排序，不自动采用或按年份换模型。2025年24个月性能NA。1000次公司级配对区间以固定预测为条件，未覆盖局部寻优偏差。

```bash
python -m experiments.post_st_tree_count_local_v1.prepare
# 完整pytest含独立R，通过后提交冻结协议和运行代码
python -m experiments.post_st_tree_count_local_v1.run freeze
python -m experiments.post_st_tree_count_local_v1.run pilot
python -m experiments.post_st_tree_count_local_v1.run formal
python -m experiments.post_st_tree_count_local_v1.analyze metrics
python -m experiments.post_st_tree_count_local_v1.analyze functions
python -m experiments.post_st_tree_count_local_v1.audit
# 阅读全体汇总，编写INTERPRETATION_ZH.md，再生成报告
python -m experiments.post_st_tree_count_local_v1.report
```

使用run_location.json指向的新唯一私有目录，既有阶段不可覆盖重跑。不可向已发布task目录直接写入。公司标识、数据、模型、预测与函数表均留私有目录；公开目录只放汇总、代码和哈希。保留CURRENT_POST_ST_MODEL.json、主模型文件夹和此前全部实验。
