# 每树候选变量数10—35实验

固定当前74项C039、17/503/22各32树、q2、深度4、退市支持保护、共享切点和父节点收缩10，只变每树候选池大小。全整数26值、历史2020—2022和外层2023—2025，每组实际重新训练三棵32树森林，合计468次拟合。现有主模型保持不变。

这是已暴露队列上的探索性分析。历史前向选值先锁定，外层逐值比较与网格极值另报，不能据此宣称总体最优。

## 复现

使用run_location.json定位唯一私有运行目录；重做时须新建私有目录并更新定位文件，不覆盖旧运行。训练输入和公司模型/预测不进入Git。

```bash
python -m experiments.post_st_feature_pool_grid_v1.prepare
# 完整测试并提交协议和代码后
python -m experiments.post_st_feature_pool_grid_v1.schedule
hetero-run PRIVATE_RUN/pilot.jsonl
# 核对成功与数值后
hetero-run PRIVATE_RUN/formal.jsonl
python -m experiments.post_st_feature_pool_grid_v1.analyze history
# 提交历史选择结果后
python -m experiments.post_st_feature_pool_grid_v1.analyze outer
python -m experiments.post_st_feature_pool_grid_v1.report
```

两个节点各自环境由launch.py确定。全部任务单线程，输入只读共享NFS，输出由调度器核验回传。配置、输入哈希和实际任务清单均记录；失败不自动重拟合。

主评价12个月退市；24个月和恢复为次要。2025年24个月性能NA。完整展开覆盖全部训练/评价行、原生时间和两原因，最大容差1e-10。模型使用各年训练边际乘积定义函数参考分布，相关特征下不构成因果归因或统计可辨识性证明。
