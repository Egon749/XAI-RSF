# 已保存种子的等权预测组合

固定比较原96树（42/101/202）、指定七种子224树（1009/503/17/202/101/419/300）和加入42的八种子256树。每家企业、时点、原因先平均各32树成员的CIF，再重新计算性能；不是平均AUC/AP。只用已经收缩的74输入支持保护模型，不重训、不再收缩或校准、不搜索权重、不更换参考。

```bash
python -m experiments.post_st_seed_ensemble_v1.prepare
# 完整pytest（含独立R）通过后，提交代码与冻结协议。
python -m experiments.post_st_seed_ensemble_v1.run freeze
python -m experiments.post_st_seed_ensemble_v1.run exports
python -m experiments.post_st_seed_ensemble_v1.analyze
python -m experiments.post_st_seed_ensemble_v1.report
```

六个独立合并/导出任务使用现有两节点Slurm与NFS。run_location.json指向新的唯一私有目录。公司标识、输入、模型、函数值与逐公司预测仅在服务器；公开仅代码、协议、汇总与哈希。2025年24个月性能NA，合并24个月仅包含2023/2024。1000次固定预测配对公司重抽样沿用原索引。

名单是在展示种子排名后指定的；本次是探索性组合评价。224/256与96树预算不同，不能把差异单独归于种子质量或声称为独立确认。原96树、CURRENT_POST_ST_MODEL.json、主模型文件夹、历史工件均保留。
