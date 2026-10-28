# 冻结事件与公告依据审计

状态：**PROVISIONAL / POST_EXPOSURE_EXPLORATORY**。本轮新增训练、校准及阈值选择均为 **0**。

323个独立事件全部纳入查询和逐事件证据表，没有删样本。242个事件取得候选PDF；排除债券受托报告、区分保荐/法律意见后，**239个实际实施原因有官方公告证据确认，84个原因仍未决**。其中评价期64个确认、51个未决。VERIFIED仅指实际实施原因及所引条款确认；不是整套法律适用、全部历史版本及原始财报的认证。239个中仍有198个版次字段未闭合；其余有引用版次也不自动意味着所有例外核验完成。

确认的多重触发33个，确认适用财政年度/过渡安排24个。两项标志可重叠；追溯重述是事实/时钟标志，不独立算一个风险警示触发族。没有把公告中“未来可能终止上市”的模板条款当成当前触发。逐事件摘要及一事件多条证据长表均在私有目录。

**标签复核候选3个，均为训练期，合计12条已标目标窗口。** 其中2个有公司正式实施公告：冻结原因码02“资产缩水”，实际公告为资金占用普通ST，涉及8条训练窗口；建议复核为竞争事件。另1个冻结为“两年亏损”，当期保荐意见指停产/银行冻结，涉及4条，缺正式实施公告，证据级别较低。未应用任何修订，不声称3个都已证实标错，更不据此推断模型性能变化。

| candidate_type | original_label | proposed_classification | independent_events | affected_rows | affected_role | evidence_status | action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target_to_competing_review | 1 | 2 | 1 | 4 | train | SPONSOR_OPINION_ONLY_UNRESOLVED | NOT_APPLIED |
| target_to_competing_review | 1 | 2 | 2 | 8 | train | OFFICIAL_IMPLEMENTATION_REASON_CONFLICT | NOT_APPLIED |

原公告日与实施日独立保留；来源公开时间另列，不替换主事件时钟。公开日期能匹配240个事件的冻结公告日，1个有日期差异，82个无符合筛选的官方PDF；这些是来源覆盖统计，不等于全部日期获PIT认证。后续公告重述历史、文件内部笔误与引用旧版本分别留审计备注，不以此自动改事件日。原结局仍有713条评价未知记录。

| role | frozen_event_type | board | verification_status | independent_events | quarter_rows | multiple_triggers | transition_applies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eval | 1 | SSE_MAIN | UNRESOLVED | 6 | 17 | 0 | 0 |
| eval | 1 | SSE_MAIN | VERIFIED | 18 | 53 | 5 | 5 |
| eval | 1 | SSE_STAR | UNRESOLVED | 3 | 7 | 0 | 0 |
| eval | 1 | SSE_STAR | VERIFIED | 2 | 7 | 0 | 2 |
| eval | 1 | SZSE_CHINEXT | UNRESOLVED | 10 | 27 | 0 | 0 |
| eval | 1 | SZSE_CHINEXT | VERIFIED | 10 | 29 | 6 | 3 |
| eval | 1 | SZSE_MAIN | UNRESOLVED | 13 | 37 | 0 | 0 |
| eval | 1 | SZSE_MAIN | VERIFIED | 17 | 46 | 7 | 7 |
| eval | 2 | SSE_MAIN | VERIFIED | 8 | 16 | 0 | 0 |
| eval | 2 | SSE_STAR | VERIFIED | 1 | 2 | 0 | 0 |
| eval | 2 | SZSE_CHINEXT | UNRESOLVED | 11 | 36 | 0 | 0 |
| eval | 2 | SZSE_CHINEXT | VERIFIED | 5 | 13 | 0 | 0 |
| eval | 2 | SZSE_MAIN | UNRESOLVED | 8 | 23 | 0 | 0 |
| eval | 2 | SZSE_MAIN | VERIFIED | 3 | 9 | 0 | 1 |
| train | 1 | SSE_MAIN | UNRESOLVED | 15 | 53 | 0 | 0 |
| train | 1 | SSE_MAIN | VERIFIED | 53 | 203 | 7 | 1 |
| train | 1 | SZSE_CHINEXT | UNRESOLVED | 2 | 4 | 0 | 0 |
| train | 1 | SZSE_CHINEXT | VERIFIED | 4 | 11 | 0 | 1 |
| train | 1 | SZSE_MAIN | UNRESOLVED | 7 | 24 | 0 | 0 |
| train | 1 | SZSE_MAIN | VERIFIED | 96 | 359 | 8 | 2 |
| train | 2 | SSE_MAIN | UNRESOLVED | 4 | 13 | 0 | 0 |
| train | 2 | SSE_MAIN | VERIFIED | 6 | 24 | 0 | 0 |
| train | 2 | SZSE_CHINEXT | UNRESOLVED | 2 | 8 | 0 | 0 |
| train | 2 | SZSE_CHINEXT | VERIFIED | 4 | 15 | 0 | 2 |
| train | 2 | SZSE_MAIN | UNRESOLVED | 3 | 9 | 0 | 0 |
| train | 2 | SZSE_MAIN | VERIFIED | 12 | 48 | 0 | 0 |
