# 固定森林的历史仿射校准与跨期参数收缩

本轮只拟合校准器，禁止调用森林fit。保留原83变量q2深度4＋直接父节点AJ10森林和CURRENT_POST_ST_MODEL.json。三组：原预测baseline、逐期独立校准independent、跨期参数收缩temporal。后两组使用完全相同的历史样本外预测、成熟队列、损失、概率约束和稀少事件规则；唯一新增项为向上一期temporal参数的惩罚。

每个原因校准样本中独立事件公司不足10家时固定斜率1；不是把bootstrap重复记录当新增公司。该门槛不是功效保证。身份先验20个等效公司，跨期参数惩罚强度也固定20，不进行超参数搜索。初始一期无跨期项。窗口重叠不产生独立证据，跨期收缩可能拖慢真实漂移响应。

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
