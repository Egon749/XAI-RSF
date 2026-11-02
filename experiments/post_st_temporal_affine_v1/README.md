# 固定森林的历史仿射校准与跨期参数收缩

本轮只拟合校准器，禁止调用森林fit。保留原83变量q2深度4＋直接父节点AJ10森林和CURRENT_POST_ST_MODEL.json。三组：原预测baseline、逐期独立校准independent、跨期参数收缩temporal。后两组使用完全相同的历史样本外预测、成熟队列、损失、概率约束和稀少事件规则；唯一新增项为向上一期temporal参数的惩罚。

每个原因校准样本中独立事件公司不足10家时固定斜率1；不是把bootstrap重复记录当新增公司。该门槛不是功效保证。向不校准状态收缩的强度固定20，跨期参数惩罚强度也固定20，不进行超参数搜索。初始一期无跨期项。窗口重叠不产生独立证据，跨期收缩可能拖慢真实漂移响应。

12/24个月分别产生全曲线仿射映射，每一条都具有全输入概率证书和精确q2函数；它们并非同一个联合校准模型。另行报告12月端点与24月端点的相容性，不通过裁剪制造一致。2025年24月可导出和验证预测，但结局性能仍NA。

正式18个校准器（12个外层、6个初始化）；50次配对公司重抽样各重跑两条完整链，共900个校准器。森林与OOF预测固定，因此只衡量校准估计不确定性，不是整个建模流程的不确定性。固定点预测另做1000次按测试年分层的配对评价bootstrap。

```bash
python -m experiments.post_st_temporal_affine_v1.prepare
# 完整测试含独立R，提交冻结协议及代码
python -m experiments.post_st_temporal_affine_v1.run freeze
python -m experiments.post_st_temporal_affine_v1.run point
python -m experiments.post_st_temporal_affine_v1.run bootstrap
```

prepare/freeze仅用于全新目录。服务器私有目录见run_location.json。pool输入只含测试X和ID，测试结局放在单独evaluation_input；校准历史结局按各自截止点截断。企业、参数、函数值、预测保留服务器，代码和汇总报告入库。


已完成的正式结果见[RESULTS_ZH.md](RESULTS_ZH.md)，点估计与重抽样均采用修订后的solver_v2。协议冻结提交052ce4eb91d6c7b554362a56934a23753ff307a4；求解器修订快照06b62a92f9adb1b36fed50821e97a1e04bdc32c7。数学目标、惩罚、抽样与精度门槛未变；首轮失败及所有成功记录保留。

数值修订执行顺序（现有输出禁止覆盖；这些计算已完成，不要原地重复执行）：

```bash
python -m experiments.post_st_temporal_affine_v1.run freeze --revision solver_v2
python -m experiments.post_st_temporal_affine_v1.run point --revision solver_v2
python -m experiments.post_st_temporal_affine_v1.run bootstrap --revision solver_v2
python -m experiments.post_st_temporal_affine_v1.run activate --revision solver_v2
python -m experiments.post_st_temporal_affine_v1.analyze metrics
python -m experiments.post_st_temporal_affine_v1.analyze stability
python -m experiments.post_st_temporal_affine_v1.audit
python -m experiments.post_st_temporal_affine_v1.deliver
```

分析只读拟合工件，将汇总写入独立reports；JSON审计有存在保护。复现完整实验需先在common.py指定新的唯一RUN目录，遵循prepare→全测试→干净提交→冻结代码→点估计→重抽样的顺序；audit.py的实际执行台账针对本轮已保存的四批作业，复现时应记录新作业而非借用本轮次数。不得为复现报告覆盖本次输出。

目录包含48行年度/合并两原因指标、36行原森林单种子指标、918个校准器诊断及哈希、配对95%区间、1200行重抽样评价、完整数值验证、计算成本、全部204条执行记录及两张PNG/PDF图。单种子原预测仅供参考；校准器作用于三种子均值，不将集成概率证书套到单棵树或单种子。

本轮完整计算保留原森林，年内排序不变，但跨年排序和校准估计波动均有实际权衡；不将“完整解释仍精确”解释为学到了真实经济因果作用。CURRENT_POST_ST_MODEL.json未修改。
