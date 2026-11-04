# ST 后退市原因与经济解释审计

固定读取基准提交 f01e07e8855787d8832d59d33128e6b66cc11f10 的 C039（17、503、22）。协议提交 f448a6a，首次盲审登记与类别计数锁定于 19a0c90。后续补充核对了风险提示的正文及第二次提示回引，没有改变类别、案例集合或比较规则。执行不调用拟合接口。

- [中文研究报告](RESULTS_ZH.md) 与 [论文表述及决定](METHOD_DECISION_ZH.md)。
- [协议](FROZEN_PROTOCOL.json)、[事件定义](EVENT_DEFINITION_AUDIT.json)、[规则版本](DELISTING_RULE_VERSION_REGISTRY.csv)、[分类审计](CAUSE_CLASSIFICATION_AUDIT.csv)、[年度计数](CAUSE_COUNTS_BY_YEAR.csv)、[证据覆盖](EVIDENCE_COVERAGE.csv)。
- [全模型哈希与重构](MODEL_INTEGRITY_AUDIT.json)、[事件原文覆盖与私有登记摘要](SOURCE_AND_HASH_AUDIT.json)、[额外输入与对齐](ALIGNMENT_CHECKS.json)、[执行回执](execution_receipt.json)。
- CAUSE_SIGNAL_COMPARISON 的 main_ 为原始主效应，alloc_ 为主效应加涉及该变量的每个交互的一半。CIF 和贡献使用百分点；门槛距离以倍数单列。price_floor_ratio 是上一可得收盘价相对 1 元的倍数。market_cap_floor_ratio 使用总市值和原点适用的板块门槛，仅是单日比较。mean_abs 先在每家公司完成种子平均、相加和分摊，再取绝对值。
- TRADING_VS_NONTRADING、CAUSE_TOP20_CAPTURE、JOINT_SUPPORT、CAUSE_EXPLANATION_STABILITY、FIXED_INFORMATION_SET_CAPTURE、CHANNEL_CONCORDANCE_ACCOUNTING 分别给出跨类对照、年度名额捕获、联合支持、算法波动、既有四格模型及配对排序构成；不足的单元保留 NA。CAUSE_TOP20_CAPTURE 中 C2_pure 排除同类双触发，nontrading 与 C1 在 24 个月是同一批公司。分窗结果不能相加。
- ORIGIN_AUDIT_EVIDENCE、AUDIT_FLAG_STATES、SUPPORTED_RAW_PROFILES、PRICE_FUNCTION_BY_TRAINING_TERTILE 与 PRIOR_EVIDENCE_RECONCILIATION 区分输入时点核对、原始函数形状、归因与最终条款。图 6 只使用通过训练格和评价人数保护的分位组。图形目录包括六张 PDF 与对应 PNG。

## 服务器复核顺序

从本分支仓库根目录运行。设置 CAUSE_AUDIT_RUN 为仅研究者可访问的新建输出目录；CAUSE_POLICY_RUN 指向法规及证券索引归档；CAUSE_CAP_DICTIONARY 指向包含 Dsmvtll 千元单位的交易数据字典。服务器内的源数据目录按原实验的只读入口解析。Git 需能读取政策分支提交 847619a 和四信息集对照提交 614eb0b。FOREST/R 检验使用该仓库已配置的研究 Python 和 R 环境，官方网页收集另需 requests、BeautifulSoup、pdftotext。

```bash
m=experiments.post_st_delisting_cause_audit_v1
python -m "$m.inventory"
python -m "$m.integrity"
python3 -m "$m.rules"
python3 -m "$m.collect_decisions"
python3 -m "$m.filings"
python -m "$m.classify"
python -m "$m.signals"
python -m "$m.profiles"
python -m "$m.concordance"
python -m "$m.checks"
python -m "$m.figures"
python -m pytest -q
```

证据检索和人工核读为必要输入。仅在私有目录保存按证券代码组织的 discovery 索引（code、历史名称 name、官方候选 title/url）、官方原件及检索失败记录、以及 blind_review_PRIVATE.csv（code、category、grounds）。应当先在看不到预测的情况下阅读原文、记录法律原因及所有依据，再独立按实体条款章节复核。重放时可从该次私有审计归档复制这两个输入及原文缓存，无需重新联网。若重新检索，保留新的时间戳、原件哈希和版次。classify 导出 DELISTING_CAUSE_REGISTRY_PRIVATE.csv，仅供服务器私下复核。不得从公开统计重建个案登记。所有公开图表均可从本目录汇总表重画。

法定摘牌日可由同一公司正式摘牌公告确认；决定日在有冲突时优先采交易所原文。对没有标注首次、也未由后续同系列公告明确回指的提示，另存“最早发现”日期。未把数据库的执行日期自动当成法律证据。

不发布逐公司预测、具体公司解释、官网个案 URL 或服务器位置。公开的 SOURCE_AND_HASH_AUDIT 中只保留登记、查验文件的哈希和计数。未将先前可能存在的个案发布授权扩展到本实验。
