# 风险指数：已冻结、因历史分母不可定义而停止

请先读 [RESULTS_ZH.md](RESULTS_ZH.md)。本目录不是已完成的分级验证结果：实际完成了参考O/E验收和2019年历史来源/配置/预测审计，随后触发停止。所有未执行指标均为NA并写明原因，真实分档图未生成。

来源分支/提交为calendar-level `12376af1ed01bab747814bfda1cfa3a7fe6cd26f`；协议提交为 `d8880611b40f22f641e25219839c93485d3aa2e7`。只读取未做校准或水平更新的cause-support baseline收缩预测。

在仓库环境重现审计，必须指定新的私有目录：

```bash
RISK_INDEX_RUN=/绝对路径/新的私有目录 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/oliver/.venvs/xai-baselines-20261027/bin/python \
  -m experiments.post_st_risk_index_v1.audit
```

真实输入位置见run_location.json。`audit/STOP.json`表示协议性停止，不是完成；只有所有历史条件通过才会生成PASS.json。已有audit子目录不可重用。停止不自动删除历史样本、不重训、不平滑分母，也不启动后续评价。

`index.py`是固定公式与阈值分档函数，完全不接受结局；`tests/test_risk_index.py`仅用虚构数据检验基线非100、分解、可逆、不截断、尺度不变性、阈值方向、并列和支持不足。通过这些测试不等于真实企业实验通过。

所有真实基准μ、阈值、公司标识、预测、指数和函数参数禁止公开。只有审计状态、聚合O/E、哈希与未执行表在本目录。既有rankings发布授权不扩展。本次小规模审计直接执行，未提交Slurm任务，未拟合森林或校准器。

完整测试（含独立R）的执行日志为TESTS_full.log；使用既有CRFOREST_RSCRIPT和EW_RLIB环境。USER_GUIDE_ZH.md说明指数能与不能表示什么；FICTIONAL_EXAMPLE_ZH.md完全虚构；PAPER_NOTE_ZH.md仅提供定义与真实停止结果，不声称完成尚未执行的实证检查。
