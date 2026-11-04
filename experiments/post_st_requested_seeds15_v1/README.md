# 15个指定种子的退市和恢复预测

每个年度模型使用各年1月1日前的信息。每个种子32树，按企业等权平均15个种子的预测得到480树组合。先核验98、17、503、5001、2570、22的18个已存森林，再用88、75、122、105、155、178、5003、722、528实际训练27个森林。2024年种子88试运行使用完整32树，通过数值与内存检查后计入正式网格。

沿用74变量、退市支持保护、q≤2、深度4、共享切点、均衡25变量和强度10的直接父节点AJ收缩。比较2023—2025年有充分随访的12/24个日历月结果，统计两个原因、原始与收缩预测、按年度取名额的Top20、合并指标和原1000次按年度分层的公司配对区间。2025年24个月性能为NA。

私有数据、模型和企业预测存放位置见 run_location.json。独立工作目录先运行 prepare 和含独立R的完整 pytest，提交协议后按以下顺序执行。Slurm逐任务验证模型输出和回传，实际CPU分配见 CPU_PLACEMENT.json。

```bash
python -m experiments.post_st_requested_seeds15_v1.prepare
# 测试通过且代码与协议已提交
for stage in freeze pilot formal stage_ensemble ensemble; do
  python -m experiments.post_st_requested_seeds15_v1.run "$stage" || exit 1
done
python -m experiments.post_st_requested_seeds15_v1.analyze
python -m experiments.post_st_requested_seeds15_v1.cpu_placement
# 核对失败记录、写入结果解读后生成年度表及图
python -m experiments.post_st_requested_seeds15_v1.report
python -m experiments.post_st_requested_seeds15_v1.publication_check
```

本轮种子在已有年度结果展示后指定，按探索性比较解释。现有96树组合作为参照。
