# 私有完整交付索引

根目录：`/home/oliver/MainDiskData/xai_rsf_transparent_cif_v2_20261028_p3k845e7`。

- `results/formal/2546/task-0/output`、task-1、task-2依次为42、101、202。各自fit下六个方案均含functions_PRIVATE.npz、manifest_PRIVATE.json、integrity.json、complete_tables_PRIVATE.json、complete_cases_PRIVATE.json、complexity_PRIVATE.json。三树库、完整候选日志、全局权重、EBM模型与每轮分数全部保留。
- `model_audit/*_all_cells_PRIVATE.json`：18份全函数表，含所有缺失格、记录/公司组/原因别独立事件支持；`*_complete_cases_PRIVATE.json`：18份、每份6个固定匿名案例，完整日历3/6/12个月解释。真实记录对应关系仅在私有文件中。
- `standalone_deployment/`：可单独执行的完整透明函数运行包；verify.py不依赖原树代码，deployment_receipt.json记录18模型全体23799记录的核验。主程序 `python -I verify.py`；如再次核验，先复制到新目录或指定新收据，不覆盖此次证明。
- `results/formal/2546/task-*/output/predictions`：全体私人预测，所有seed和model均保留；`evaluation`：统一结果。
- `results/formal/2546/task-3/output/demo`：27组模拟、9次可部署权重QP及9次oracle容量QP收据。
- `results/bootstrap/2547`：10批共同公司抽样；`bootstrap_replicates_PRIVATE.npz`：1000次所有模型/种子/年度结果。
- `audit/`：代码归档、输入哈希、测试日志、冻结与发布核验；formal.jsonl.results-2546.json及bootstrap.jsonl.results-2547.json为原调度运行收据。

Git仅提交代码、字段定义、冻结协议、汇总结果及模型文件SHA256。真实企业明细、预测、参数、原始切点与个体解释不上传。
