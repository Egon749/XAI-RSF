# 内生解释与可审计性

低阶模型完整函数展开直接等于森林CIF，独立函数求值不遍历森林。以下包含全部函数，不是TopK：

| fold | variant | main_functions | interaction_functions | spatial_table_cells | scalar_table_cells | factored_leaf_coefficients | low_support_cell_fraction | zero_support_cell_fraction | reconstruction_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V19 | depth2 | 18 | 37 | 520 | 212160 | 52224 | 0.248077 | 0.188462 | 3.33067e-16 |
| V19 | q2 | 18 | 38 | 8353 | 3.40802e+06 | 388008 | 0.474321 | 0.235724 | 2.77556e-16 |
| V19 | q1 | 6 | 0 | 902 | 368016 | 386376 | 0.111973 | 0 | 2.22045e-16 |
| V20 | depth2 | 16 | 35 | 455 | 219310 | 61696 | 0.226374 | 0.186813 | 3.33067e-16 |
| V20 | q2 | 16 | 36 | 8521 | 4.10712e+06 | 455972 | 0.373782 | 0.161835 | 4.44089e-16 |
| V20 | q1 | 6 | 0 | 947 | 456454 | 477180 | 0.127772 | 0 | 2.77556e-16 |

空间格数含缺失状态且不合并阈值；标量表规模再乘原生时间节点与两个原因。保存采用无损矩形因子形式，压缩存储不等于较少的逻辑分区。unrestricted可能高于二阶，拒绝导出伪完整二阶解释。

| model | main_jaccard | pair_jaccard | shared_mains | shared_pairs | note |
| --- | --- | --- | --- | --- | --- |
| depth2 | 0.789474 | 0.411765 | 15 | 21 | descriptive across different calendar folds, not independent training uncertainty |
| q2 | 0.789474 | 0.396226 | 15 | 21 | descriptive across different calendar folds, not independent training uncertainty |
| q1 | 0.714286 |  | 5 | 0 | descriptive across different calendar folds, not independent training uncertainty |

component_agreement.csv用共同V19训练参考的512个固定训练点描述两折函数值相关及非零符号一致性，不是完整分区的单调方向定理，也不是重复训练稳定性置信区间。跨年份样本不同，不能把差异全归因于模型不稳定。

低支持/零支持格比例来自训练记录数，不是独立公司事件信息矩阵。因相关变量导致的无支持组合仍存在。函数规范化、精确重构、可读复杂度、统计可靠性是四个不同性质。全部真实匿名案例含所有主效应/交互及最终CIF，留在服务器cases_PRIVATE.json。贡献为概率尺度、依赖参考分布，非因果。
