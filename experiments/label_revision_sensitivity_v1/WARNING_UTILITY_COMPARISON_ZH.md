# 预警效用与配对名单流转

Top20为主要季度预算，Top25为扩展预算；全部23,799行参加各季度排序，未知结果保留核查负担。企业级命中占比=有已确认目标窗口命中的去重公司/被核查去重公司，不是记录级命中率；未知不能当误报。下表三seed概率集成，跨年目标事件去重79个；可提前3/6个月机会均为60个，年度分别25/35。

| scope | model | top_fraction | alert_company_groups | confirmed_hit_company_groups | company_hit_fraction | detected_events | independent_events | announcement_detected_3m | announcement_detected_6m | unknown_outcome_alerts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | L0_q2 | 0.2 | 928 | 37 | 0.0398707 | 37 | 44 | 20 | 18 | 60 |
| 2023 | L0_q2 | 0.25 | 1117 | 41 | 0.0367055 | 41 | 44 | 23 | 20 | 67 |
| 2023 | L0_unrestricted | 0.2 | 930 | 41 | 0.044086 | 41 | 44 | 22 | 21 | 66 |
| 2023 | L0_unrestricted | 0.25 | 1147 | 41 | 0.0357454 | 41 | 44 | 22 | 22 | 74 |
| 2023 | L1_q2 | 0.2 | 927 | 37 | 0.0399137 | 37 | 44 | 20 | 18 | 60 |
| 2023 | L1_q2 | 0.25 | 1114 | 41 | 0.0368043 | 41 | 44 | 23 | 21 | 68 |
| 2023 | L1_unrestricted | 0.2 | 933 | 41 | 0.0439443 | 41 | 44 | 22 | 22 | 64 |
| 2023 | L1_unrestricted | 0.25 | 1149 | 42 | 0.0365535 | 42 | 44 | 23 | 22 | 75 |
| 2023 | M2_allsector_reference | 0.2 | 919 | 39 | 0.0424374 | 39 | 44 | 22 | 21 | 62 |
| 2023 | M2_allsector_reference | 0.25 | 1132 | 40 | 0.0353357 | 40 | 44 | 22 | 22 | 70 |
| 2024 | L0_q2 | 0.2 | 954 | 43 | 0.0450734 | 43 | 60 | 25 | 24 | 157 |
| 2024 | L0_q2 | 0.25 | 1138 | 49 | 0.043058 | 49 | 60 | 26 | 26 | 196 |
| 2024 | L0_unrestricted | 0.2 | 938 | 51 | 0.054371 | 51 | 60 | 29 | 29 | 172 |
| 2024 | L0_unrestricted | 0.25 | 1131 | 54 | 0.0477454 | 54 | 60 | 31 | 31 | 197 |
| 2024 | L1_q2 | 0.2 | 953 | 42 | 0.0440714 | 42 | 60 | 24 | 23 | 157 |
| 2024 | L1_q2 | 0.25 | 1138 | 49 | 0.043058 | 49 | 60 | 26 | 26 | 196 |
| 2024 | L1_unrestricted | 0.2 | 940 | 51 | 0.0542553 | 51 | 60 | 29 | 29 | 172 |
| 2024 | L1_unrestricted | 0.25 | 1132 | 54 | 0.0477032 | 54 | 60 | 31 | 31 | 197 |
| 2024 | M2_allsector_reference | 0.2 | 935 | 47 | 0.0502674 | 47 | 60 | 27 | 26 | 168 |
| 2024 | M2_allsector_reference | 0.25 | 1130 | 51 | 0.0451327 | 51 | 60 | 28 | 28 | 202 |
| pooled | L0_q2 | 0.2 | 1171 | 62 | 0.0529462 | 62 | 79 | 45 | 42 | 217 |
| pooled | L0_q2 | 0.25 | 1379 | 67 | 0.0485859 | 67 | 79 | 49 | 46 | 263 |
| pooled | L0_unrestricted | 0.2 | 1176 | 71 | 0.0603741 | 71 | 79 | 51 | 50 | 238 |
| pooled | L0_unrestricted | 0.25 | 1413 | 73 | 0.0516631 | 73 | 79 | 53 | 53 | 271 |
| pooled | L1_q2 | 0.2 | 1169 | 61 | 0.0521814 | 61 | 79 | 44 | 41 | 217 |
| pooled | L1_q2 | 0.25 | 1373 | 67 | 0.0487983 | 67 | 79 | 49 | 47 | 264 |
| pooled | L1_unrestricted | 0.2 | 1177 | 71 | 0.0603229 | 71 | 79 | 51 | 51 | 236 |
| pooled | L1_unrestricted | 0.25 | 1416 | 73 | 0.0515537 | 73 | 79 | 54 | 53 | 272 |
| pooled | M2_allsector_reference | 0.2 | 1152 | 66 | 0.0572917 | 66 | 79 | 49 | 47 | 230 |
| pooled | M2_allsector_reference | 0.25 | 1398 | 69 | 0.0493562 | 69 | 79 | 50 | 50 | 272 |


合并Top20：q2由62/79降至61/79、核查1171→1169组，公告前3个月45→44、6个月42→41；不限阶数71/79保持，核查1176→1177组，3个月51保持、6个月50→51。修正后命中企业占比q2约5.22%、不限阶数约6.03%，仍有很大核查负担，不能用AUC较高掩盖。

Top25：q2仍67/79、核查1379→1373组，6个月46→47；不限阶数仍73/79、核查1413→1416组，3个月53→54。窗口捕获不等于及时捕获，名单规模微变不等于实际事件增益。完整逐季度记录、年度/合并去重公司、重复预警、已知无目标（其中竞争单列）及未知见warning_utility.csv。

| model | top_fraction | endpoint | common_captured | new_captured | lost_captured | common_missed | companies_entering | companies_exiting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q2 | 0.2 | window | 61 | 0 | 1 | 17 | 2 | 4 |
| q2 | 0.2 | lead3m | 44 | 0 | 1 | 15 | 2 | 4 |
| q2 | 0.2 | lead6m | 41 | 0 | 1 | 18 | 2 | 4 |
| q2 | 0.25 | window | 67 | 0 | 0 | 12 | 1 | 7 |
| q2 | 0.25 | lead3m | 49 | 0 | 0 | 11 | 1 | 7 |
| q2 | 0.25 | lead6m | 46 | 1 | 0 | 13 | 1 | 7 |
| unrestricted | 0.2 | window | 71 | 0 | 0 | 8 | 12 | 11 |
| unrestricted | 0.2 | lead3m | 51 | 0 | 0 | 9 | 12 | 11 |
| unrestricted | 0.2 | lead6m | 50 | 1 | 0 | 9 | 12 | 11 |
| unrestricted | 0.25 | window | 73 | 0 | 0 | 6 | 13 | 10 |
| unrestricted | 0.25 | lead3m | 53 | 1 | 0 | 6 | 13 | 10 |
| unrestricted | 0.25 | lead6m | 53 | 0 | 0 | 7 | 13 | 10 |


共同捕获/新增/丢失/共同遗漏按同一事件集合比较，不在内控或任何政策子集重排名单。公司状态可能跨季度重叠，未知/竞争/已知无目标的公司数不能简单相加；记录类别则逐项对账等于全部预警记录。

| contrast | metric | point_difference | conditional_ci_lower | conditional_ci_upper |
| --- | --- | --- | --- | --- |
| q2_L1_minus_L0 | company_hit_fraction | -0.000764848 | -0.00253998 | 0.000179636 |
| q2_L1_minus_L0 | event_coverage | -0.0126582 | -0.0422686 | 0 |
| q2_L1_minus_L0 | lead3m_detected | -1 | -3 | 0 |
| q2_L1_minus_L0 | lead6m_detected | -1 | -3 | 0 |
| unrestricted_L1_minus_L0 | company_hit_fraction | -5.12949e-05 | -0.000574129 | 0.000428072 |
| unrestricted_L1_minus_L0 | event_coverage | 0 | 0 | 0 |
| unrestricted_L1_minus_L0 | lead3m_detected | 0 | 0 | 0 |
| unrestricted_L1_minus_L0 | lead6m_detected | 1 | 0 | 3 |
| L0_q2_minus_unrestricted | company_hit_fraction | -0.00742795 | -0.0138052 | -0.00153135 |
| L0_q2_minus_unrestricted | event_coverage | -0.113924 | -0.204552 | -0.0298398 |
| L0_q2_minus_unrestricted | lead3m_detected | -6 | -12 | 0 |
| L0_q2_minus_unrestricted | lead6m_detected | -8 | -14.025 | -2 |
| L1_q2_minus_unrestricted | company_hit_fraction | -0.0081415 | -0.0144494 | -0.00241548 |
| L1_q2_minus_unrestricted | event_coverage | -0.126582 | -0.2125 | -0.0454545 |
| L1_q2_minus_unrestricted | lead3m_detected | -7 | -13 | -2 |
| L1_q2_minus_unrestricted | lead6m_detected | -10 | -16 | -4 |
| gap_change | company_hit_fraction | -0.000713553 | -0.00266115 | 0.000530176 |
| gap_change | event_coverage | -0.0126582 | -0.0422686 | 0 |
| gap_change | lead3m_detected | -1 | -3 | 0 |
| gap_change | lead6m_detected | -2 | -5 | 0 |


1000共同公司bootstrap保持完整季度名单固定，估计对象是冻结名单在公司构成扰动下的效用；不重新排名。相同捕获事件集合可产生[0,0]的条件差值区间，不是总体部署无差异的证明。
