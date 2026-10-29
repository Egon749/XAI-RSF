# 完整低阶函数保留，稳定性不能由精确性推出

三个32树模型的平均等价于完整96树的代数合并，未额外拟合。所有443条训练／测试记录、全部164个原生事件及显示时点均重构并重载核验，最大误差1.332e-15。

| trees | main_functions | interaction_functions | spatial_table_cells | scalar_table_cells | factored_leaf_coefficients | low_support_cell_fraction | zero_support_cell_fraction | support_threshold | reconstruction_error | reload_error |
|---|---|---|---|---|---|---|---|---|---|---|
| 96 | 64 | 147 | 3194 | 1034856 | 331128 | 0.386349 | 0.236381 | 10 | 0.000000 | 0.000000 |


函数是各原因CIF的主效应与二元交互贡献，全部相加还原预测；不是风险率贡献，也不是因果影响。q2限制在训练时执行，父节点混合不改变分裂几何，但精确分解不是本方法独有的性质。

| seed_a | seed_b | order | selected_a | selected_b | common | jaccard | interpretation |
|---|---|---|---|---|---|---|---|
| 42 | 101 | 1 | 42 | 41 | 28 | 0.509091 | selection overlap only, not proof of stable function shape |
| 42 | 101 | 2 | 53 | 57 | 9 | 0.089109 | selection overlap only, not proof of stable function shape |
| 42 | 202 | 1 | 42 | 46 | 30 | 0.517241 | selection overlap only, not proof of stable function shape |
| 42 | 202 | 2 | 53 | 61 | 10 | 0.096154 | selection overlap only, not proof of stable function shape |
| 101 | 202 | 1 | 41 | 46 | 29 | 0.500000 | selection overlap only, not proof of stable function shape |
| 101 | 202 | 2 | 57 | 61 | 11 | 0.102804 | selection overlap only, not proof of stable function shape |


这里只检验三个种子的函数选择重合，不等于证明形状稳定、统计可辨识性或参数可靠。大量主效应和交互仍可能难以阅读。

| seed | nodes | leaves | zero_target_leaves | at_most_one_target_leaves | zero_target_leaves_with_borrowed_risk | max_independent_leaf_error | raw_and_shrunk_geometry_identical |
|---|---|---|---|---|---|---|---|
| 42 | 678 | 355 | 128 | 188 | 46 | 0.000000 | True |
| 101 | 640 | 336 | 119 | 171 | 48 | 0.000000 | True |
| 202 | 630 | 331 | 110 | 159 | 30 | 0.000000 | True |


没有本叶退市事件的叶可以通过父节点借入正概率，但本叶事件支持仍为零。完整函数、切点、模型及逐企业贡献保留在私有ensemble_private和任务目录。
