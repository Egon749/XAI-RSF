# 面向完整透明预测器的代码更新

服务器2026-10-28已完成本轮真实运行：[完整执行交付与研究判断](experiments/transparent_cif_v2/server/results_20261028/SERVER_EXECUTION_ZH.md)。99项完整测试无跳过，18个透明CIF模型及6个EBM完成；透明部署通过，但联合候选＋全局权重未获得独立预测增益支持。下文及原validation目录保留代码提交时的历史验证范围。

入口：[新模块、冻结配置及Codex运行说明](experiments/transparent_cif_v2/README_ZH.md)。

本更新基于`5de9cf55a40de6b8af41a6877e51ab3529c6c003`，只增加可选代码，不覆盖旧森林/函数编译器、原配置或任何历史结果。重点是保留“完整原生CIF主效应＋二元交互”的预测器，而不是新增黑箱或SHAP代理。

1. 联合成对候选树允许纯交互进入；保留公司bootstrap与Aalen–Johansen叶估计。
2. 公司组隔离的内部留出资料学习一套全局非负树权重，对所有企业、时间和原因相同。
3. 原树银行、同树数普通树扩展、配对扩展分别作等权/加权消融。
4. 最终模型只靠完整函数展开即可预测；实际所有分量、缺失格及支持度均导出。
5. 可选EBM在历史内部公司留出集选择训练轮数，不再固定2000轮后以过拟合结果作强基线。

新API：

```python
from crforest.transparent import TransparentCIF, learn_tree_weights
from crforest.pair_candidates import augment_pair_bank, extend_ordinary_bank
```

它不是标准RSF的原配置，也不声称解决了统计可辨识性或真实命中率。树权重学习是训练步骤，编译只是精确代数转换。固定非负权重不能保证所有分离信号都恢复；模拟容量诊断专门披露这一限制。

本次实际验证及环境限制见`experiments/transparent_cif_v2/validation/`。真实企业重跑由用户的Codex在服务器执行。需要先运行包含独立R检查的完整旧＋新测试；不把当前容器缺少的依赖当成通过。
