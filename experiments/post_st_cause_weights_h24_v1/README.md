# 24个月、完整历史的原因权重匹配实验

按用户授权，将ST后首次恢复／最终退市的预测期限改为24个日历月，同时使用测试年前全部可用历史。2023测试：2012—2022入组309家，结局信息严格早于2023-01-01，14个先退市、184个先恢复；2024测试：2012—2023入组345家，信息严格早于2024-01-01，26个先退市、198个先恢复。训练删失49／44家保留，绝不改成正常。2023／2024测试36／45家，分别11／20个先退市、15／16个先恢复，24个月端点均已完整。

固定比较q2原因权重 `(退市,恢复)=(1,1),(2,1),(1,0)`，各三种子42／101／202，合计18个正式模型、2个4树试运行。原75列入组时特征、32树、深度6、叶节点支持限制、共享切点与25变量资格、bootstrap及叶内AJ均不变。核心实现直接复用前轮已经通过测试的权重接口，没有改动森林核心。本轮不使用Focal Loss、不重新校准、不加滚动记录或制度特征、不评价2025、不追加搜索。

主结果为三个种子预测先平均、再计算指标；不将种子指标均值冒充集成性能。两个年度与合并预测分别报告；合并AP重新计算。Top20%为主、Top25%补充，年度核查预算事后排序不称为在线策略。记录等权和公司等权相同，因为每家公司仅一行。2023公司进入下一折历史训练是预定滚动年度验证，所有结局仍受各自训练截止约束。

对照复用与本轮相同24个月训练／测试口径的历史无阶数限制森林及randomForestSRC预测；六个等权q2重训模型须与历史模型逐树、逐预测精确一致。它们是ST后任务的对照，不是早期首次ST预警的M2。对照仍用原因等权，不能据此声称优于同样目标加权的标准RSF。

与上轮12个月权重实验相比，历史范围和终点期限同时改变，不能把前后指标差异仅归因于期限。24个月预测本来也是不同目标；12个月外的事件进入新终点，不是给旧12个月任务凭空增加标签。

数据继承既有财务／行业历史版本及首次事件历史完整性限制，属于PROVISIONAL、已接触测试结局后的探索性研究。预测原点仍为公告日，同日财报仍未纳入，未声称完成PIT认证。企业、实际阈值、函数数值、模型参数、预测只留服务器。

固定协议见 [frozen_protocol.json](frozen_protocol.json)。新私有运行目录先建立audit/input/evaluation_input/reports/results，再执行：

```bash
python -m experiments.post_st_cause_weights_h24_v1.prepare --run "$RUN"
# 完整测试通过并提交代码后冻结：
python -m experiments.post_st_cause_weights_h24_v1.stages --stage freeze --run "$RUN" --repo "$REPO"
hetero-submit "$RUN/pilot.jsonl" physical both
python -m experiments.post_st_cause_weights_h24_v1.stages --stage pilot --run "$RUN" --job "$PILOT_JOB"
hetero-submit "$RUN/formal.jsonl" physical both
python -m experiments.post_st_cause_weights_h24_v1.stages --stage formal --run "$RUN" --job "$FORMAL_JOB"
python -m experiments.post_st_cause_weights_h24_v1.analyze --run "$RUN"
python -m experiments.post_st_cause_weights_h24_v1.audit --run "$RUN"
```

必须等待任务真正完成并通过发布门禁再进入下一步。现有两节点Slurm物理核池使用独立单线程CPU任务、NFS与结果自动回传，每任务4GiB地址空间上限。审计重放全部已保存节点／叶内统计和完整96树函数表，不重新训练。所有输出拒绝覆盖已有运行；保留旧实验及失败记录。
