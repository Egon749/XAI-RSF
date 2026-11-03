# 70—120棵统一比较，重点24个月结局

固定74变量、退市支持保护、q2、深度4、共享切点、均衡25变量、父节点AJ收缩10及24个月训练随访。三个种子42/101/202保持相同树数并等权平均；70—120范围内的有效合并树数是72、75……120，共17档。

复用81—120的全部336份raw/收缩表示，仅补充72/75/78共36个联合导出任务、72份表示。全部来自已保存森林的原编号前缀，无新拟合或校准。复用上一轮120树的数值/资源试运行。新输出全部使用独立私有目录，不覆盖已有工件。

主展示为2023/2024年24个月先退市，附12个月、恢复、raw/收缩、各种子及合并指标。2025年24个月性能NA。所有点均为已暴露样本上的探索性比较，不改写上一轮一年期排序规则，不自动更换96树参考。

```bash
python -m experiments.post_st_tree_count_70_120_v1.prepare
# 完整pytest含独立R，然后提交冻结协议和入口
python -m experiments.post_st_tree_count_70_120_v1.run freeze
python -m experiments.post_st_tree_count_70_120_v1.run formal
python -m experiments.post_st_tree_count_70_120_v1.analyze metrics
python -m experiments.post_st_tree_count_70_120_v1.analyze functions
python -m experiments.post_st_tree_count_70_120_v1.audit
# 检查结果，写INTERPRETATION_ZH.md，再生成汇总
python -m experiments.post_st_tree_count_70_120_v1.report
```

公司数据、预测、阈值和函数参数留run_location.json所指私有目录。公开仅包含汇总、版本及哈希。原数据可得日与版本限制继承；不能把未知结局当负例，不能把合并AP/AUC写成年度均值。

