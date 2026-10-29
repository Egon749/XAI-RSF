# 冻结L1上的透明CIF有限实验协议

本轮完整执行已有六方案×三个种子，检验联合成对候选与全局凸权重是否保留有效信号并改善预测/预警。数据为PROVISIONAL；2023/2024是后暴露探索性比较，不能称独立盲测。原财务披露版本PIT、行业历史时钟和47项未决事件限制保留。

算法参考fd4a87a；算法分支实际tip b6eab6a；L1交接526948f。核心core/functions/partition/pair_candidates/transparent/run/ebm_control/config逐字节等于算法参考。后续demo仅加主效应幅度与oracle收据记录；评价仅补显式版本化参考读入。实际训练SHA：3f6846db5be5272b6e54d1bb63c83f50c16420a1。最终交付提交另见Git历史，不混称训练SHA。

结构/内部留出公司角色固定role_seed=20261028、20%留出。森林参数取L1 plan森林q2，32树、深度6、每树25个声明输入、shared_quantile、mtry75，原叶支持门槛不变。75坐标的低阶约束不等于所有派生指标底层报表量都至多二阶。

固定pair:64对、最多16棵、每轴最多3切点、min_score0、seed20261028；weights:ridge0.001、max_iterations3000、gap1e-8。种子42/101/202；EBM主效应和10交互只在内部留出比较50/200/500轮，未把留出混回重训。无试运行/新增搜索。

六方案：base_equal/base_weighted、ordinary_extra_equal/ordinary_extra_weighted、pair_equal/pair_weighted。基础树完整保留，同银行等权/加权复用完全相同的树；ordinary新增数量匹配实际入选pair树。全局非负权重对企业、时间、原因全部共用。训练目标为结构资料估计删失权重的多状态Brier加固定L2，评价主表是目标原因Brier，二者不同；没有AP损失或新校准器。

拟合信息截止2022-01-01；三个seed全拟合结束、哈希冻结才预测与统一评价。训练入口设有禁止读eval元数据的守卫；原预测入口只读取X/身份/原点/期限，未读结局。评价使用原完整23,799行、原日历月提前量及未知保护。

统计规则已在训练前冻结inference_protocol.json。主口径三seed概率平均后的记录等权；公司等权及seed范围单列。主要机制比较pair_weighted−ordinary_extra_weighted，另报告三项权重作用、等权pair对ordinary、总体变化及与全训练L1 q2实用差距。所有1,000次共同公司bootstrap保持预测、名单、IPCW估计固定，保留抽样多重性。全部区间为未调整、探索性95%百分位区间；不择显著端点宣布成功，不是重训不确定性或非劣性证明。

执行入口prepare.py核验只读L1，orchestrate.py freeze/demo/train/predict/audit/bootstrap组织分阶段任务；postprocess.py调用已有评价器；collect.py汇总、deploy_and_ebm.py独立部署；report.py生成汇总。对现有完成目录不再应用补丁或重交拟合。bootstrap包装器故障的无重算恢复见recover_bootstrap.py和收据。
