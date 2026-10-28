# 旧C输入：叶节点父级AJ收缩 × 深度4有限实验

固定四组：原depth6、原树加父级AJ收缩、depth4、depth4加同样收缩。旧83列输入、样本/时钟/结局、q2、三种子42/101/202、32树/种子、共享切点及其他预算保持一致。主目标24个月先退市，12个月和恢复为补充。

收缩规则预先固定为独立入袋公司数n下的n/(n+10)，剩余权重借用直接父节点未收缩AJ。两个结局和所有时间共用权重，树结构不变；这是修改叶内估计器，非概率校准、非递归层级收缩论文的原样实现。父级数据仍来自同一树bootstrap和原训练截止，未知不当正常。

正式阶段6个depth6模型从已核验的旧基线载入并更新叶估计；6个depth4模型实际拟合，每个同时导出原AJ及收缩版。共12个计算任务、6个新森林拟合、12个叶更新、18个新模型产物；四组总计24个q2模型评价，另复用6个标准RSF。4个四树小试先检查数值与资源。每个模型保存全部函数与原始支持度，不把借来的概率当成新增事件。

在运行前冻结协议与实用保留条件。所有模型、种子及不利结果都报告，不继续试其他收缩强度、深度、权重或2025，不自动选新默认模型。历史测试结局已暴露、旧输入亦是结果暴露后由用户选择，属于探索性匹配比较。

```bash
python -m experiments.post_st_leaf_shallow_v1.prepare --stage prepare --run "$RUN"
# 完整测试及R检查通过，提交代码后：
python -m experiments.post_st_leaf_shallow_v1.prepare --stage freeze --run "$RUN" --repo "$REPO"
hetero-submit "$RUN/pilot.jsonl" p both
python -m experiments.post_st_next_open_info_v1.stages --stage pilot --run "$RUN" --job "$PILOT_JOB"
hetero-submit "$RUN/formal.jsonl" p both
python -m experiments.post_st_next_open_info_v1.stages --stage formal --run "$RUN" --job "$FORMAL_JOB"
```

使用原Slurm两机p布局、单线程独立任务、4GiB限制、NFS与校验回传，不硬编码CPU。企业预测、模型与真实切点留服务器；Git仅提交代码、冻结配置、汇总及哈希。
