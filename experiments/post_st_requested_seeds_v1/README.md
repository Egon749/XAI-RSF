# 指定八种子重跑：每种子32树

用户先指定1009、503、17、202、101、419、300，随后要求42、101、202也分别报告。去重后固定八个种子；2023、2024、2025分别实际重训，共24个raw森林、768棵树，再派生24个父节点收缩版本。先跑2024/seed300完整32树作数值/资源试运行，成功后计入正式矩阵，其余23项不重复该格。

固定74输入、q≤2、深度4、支持保护、共享切点、均衡25变量池、直接父节点AJ收缩10、各年1月1日前信息和24个日历月训练随访。保留原42/101/202各32树平均的96树参照，不创建八种子集成，不自动替换模型。原有七个种子各年结果用于检查本次实际重训的一致性。

名单在查看20种子年内12个月AUC后指定，属于暴露后探索性重跑；不是独立确认、重新抽取的无偏种子样本或最优组合证据。2025年24个月性能NA。

```bash
python -m experiments.post_st_requested_seeds_v1.prepare
# 完整pytest含独立R；提交协议与代码后：
python -m experiments.post_st_requested_seeds_v1.run freeze
python -m experiments.post_st_requested_seeds_v1.run pilot
python -m experiments.post_st_requested_seeds_v1.run formal
python -m experiments.post_st_requested_seeds_v1.analyze audit
python -m experiments.post_st_requested_seeds_v1.analyze metrics
python -m experiments.post_st_requested_seeds_v1.analyze functions
python -m experiments.post_st_requested_seeds_v1.report
```

逐年双原因/raw/收缩指标、合并指标、公司配对bootstrap、完整解释和重跑一致性诊断均保留。run_location.json指向新的唯一私有目录；公司标识、输入、真实阈值、预测、模型和函数参数仅在服务器，公开目录只放汇总、代码、配置和哈希。历史数据时钟/PIT限制保持，不更改历史输入或报告。
