# 官方规则来源与版本边界

状态：**PROVISIONAL / POST_EXPOSURE_EXPLORATORY**。本轮新增训练、校准及阈值选择均为 **0**。

登记56个来源条目（包括重复附件），成功取得52个，4个失败；不是52部独立法规。规则表共90条登记项，区分可执行财务边界与须人工处理的条款组，不代表已把所有例外编成法律裁决器。

覆盖沪深主板、原中小板所属深市主板规则族、创业板、科创板；登记2008/2012、2014、2018年4月/11月、2020年6月/12月、2022、2023年2月/8月、2024、2025相关来源。缺失版本链见 policy_evidence_issues.csv；不能跨缺口推定精确版次。原中小板没有另套主板外的一套财务门槛；板块历史身份不另行重建。

- 旧主板两年亏损与低收入单独触发，不等于2020后“利润为负且扣除后收入不足1亿元”。2024主板新口径增加利润总额并把收入门槛提高到3亿元；创业板/科创板仍为1亿元。利润须保留扣非前后关系，收入须保留扣除定义。
- 2024年4月30日正式通知明确新财务条款首个适用2024年报，2023年报沿旧标准，不能按实施公告落在4月30日后就统一使用新门槛。
- 创业板2020年6月12日通知：财务警示以FY2020起算；普通ST到2020年9月12日才实施；此前已受理的重整存在旧制例外。新增了此独立过渡节点，未把2020全年作一档。
- 科创板2024部分普通ST延后6个月；内控审计/三年亏损相关条款按通知的不同首年度处理，不套主板。分红ST自2025-01-01施行，首三年FY2022–2024，双创研发等例外保留。
- 财务审计无法表示/否定、内控审计、持续经营不确定性是不同路径；重大违法与普通虚假记载ST也分开。股票简称不能代替触发依据。

61条冻结原点（训练14、评价47）使用北交所/新三板相关代码，历史板块时钟无法在本轮确认。其B表规则候选为空、状态未决，仍在全部预测评价中。没有把沪深规则套给这些记录。

source_hash指下载原件；私有目录含对应文本、公告原件和页/条款定位。旧Word自动编号可能丢失，登记原条款并保留原件，未用OCR补造编号。6个旧Word最初按HTML误解码的问题已记录并使用antiword重新提取，初始文件保留；错误文本未用于最终分析。法律文本核验与程序测试分别记录，测试通过不等于法律语义齐备。

以下表中公司实施公告不公开，只列监管规则来源：

| source_id | status | url | sha256 |
| --- | --- | --- | --- |
| SSE_MAIN_2024_NOTICE | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20240430_10785259.shtml | 3afb22cbc8a84531713b253e1a7dfc370dbf5e515f81794b1d9667c0537466d8 |
| SSE_STAR_2024_NOTICE | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20240430_10777832.shtml | aea376ab2bef532ed4c8a7ffda973b956f0f06fcb5ae78a8cf42d184fff2eacf |
| SZSE_MAIN_2024_NOTICE | RETRIEVED | https://www.szse.cn/lawrules/rule/repeal/rules/t20240430_607069.html | 575b8546c0208e26366698655a8dfb37a78dcfb0c0eb6e566e78d5e7ea1c1f4e |
| SZSE_CHINEXT_2024_NOTICE | RETRIEVED | https://www.szse.cn/lawrules/rule/repeal/rules/t20240430_607070.html | 5e9e40980eb0bcec17582b0575bbb6237b07bddf2e8f6133996c4a3b4351d8fe |
| SSE_MAIN_2020_NOTICE | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20220107_5679292.shtml | da76611170180e9ead9f1bd15e5061ed6a27c5736fee47c406b2110e890e70cf |
| SZSE_MAIN_2020_NOTICE | RETRIEVED | https://www.szse.cn/disclosure/notice/general/t20201231_584052.html | f94a231e9dfc576ff8b5e9069633bf32d561c67992a2dc0ca557d24154433e5a |
| SZSE_MAIN_2012_NOTICE | RETRIEVED | https://www.szse.cn/disclosure/notice/company/t20120718_508731.html | 952cc661ce5886a0c4e3b0a77a444f7fc3f707bfea345ccbeaaa38c950b219bb |
| SZSE_MAIN_2012_RULE | RETRIEVED | https://docs.static.szse.cn/www/disclosure/notice/company/W020180328446393013021.pdf | 8540e16b594e2da0268b8c73e3575793fa2c9338e7c7a2dcef434a109b3ea01e |
| SSE_MAIN_2012_EXPLANATION | RETRIEVED | https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988587.shtml | ab0cac9e433cd1e24e269276221082aa658ef41d95bd73014005667baa16bded |
| SZSE_MAIN_2008_NOTICE | RETRIEVED | https://www.szse.cn/disclosure/notice/company/t20080904_508301.html | 3384f251f3aa82fa4976a98a88f44f70bccd15478edea8188e7a64979c042df0 |
| SZSE_MAIN_2014_RULE | RETRIEVED | https://docs.static.szse.cn/www/disclosure/notice/company/W020180328446480704524.pdf | f0fdb4ccbaf5cf9446e36a7dbf10d88a9d9b4341d54e7ed5ec669093b7109ffb |
| SSE_MAIN_2014_NOTICE | RETRIEVED | https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988787.shtml | c270a472de61aead6751da169ba5fc95305c2d41fb2263b1e9c7d59583eeafd0 |
| SSE_MAIN_2018_APR_NOTICE | RETRIEVED | https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20180420_4510860.shtml | e8154da235237243afb9e2a275f494c053bb5d860b6caaaa42b6fde736999749 |
| SZSE_MAIN_2018_APR_RULE | RETRIEVED | https://www.szse.cn/certificate/maind/mainrules/P020180724361500737638.pdf | 913fec025ce30a05a63f5fda2d0f7e427ab741309ecc00355eccbb072728c888 |
| SZSE_MAIN_2022_NOTICE | RETRIEVED | https://www.szse.cn/disclosure/notice/general/t20220107_590541.html | d0a128caa4272ec772a9ed4709ed59bc676de154b4fa10696de121baf9c12789 |
| SSE_MAIN_2023_FEB_NOTICE | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20230904_5726022.shtml | ddb34a881ecdb253d16de148f26a383d7d56096ebb347613d8e1551bbeb75faa |
| SZSE_MAIN_2023_AUG_NOTICE | RETRIEVED | https://www.szse.cn/www/disclosure/notice/t20230804_602477.html | e169ee60f94b24d600dafb6d77fab7970ab5fbb8c2ed48a9e6f39f9b4ce3c8bd |
| SZSE_MAIN_2023_AUG_RULE | RETRIEVED | https://docs.static.szse.cn/www/lawrules/rule/repeal/rules/W020230804625303228188.pdf | b858a84f8897bb25087062c0f464ee2bf7e47da4d3cfa54b4f0d13cec8274d1d |
| SZSE_CHINEXT_2023_AUG_NOTICE | RETRIEVED | https://www.szse.cn/lawrules/rule/repeal/rules/t20230804_602478.html | 1fdacb3d6701edc603c73efe56d27e5e26564a400ef37bfbb7cbcd36f9cc1936 |
| SZSE_CHINEXT_2012_NOTICE | RETRIEVED | https://www.szse.cn/disclosure/notice/company/t20120420_508712.html | 2af6e350d0cf89ee48d0b6985dc1d7468653b43b9f104cd43ca145081d996536 |
| SSE_MAIN_2025_NOTICE | RETRIEVED | https://www.sse.com.cn/services/listingwithsse/home/policy/supervise/c/c_20250515_10779023.shtml | 5935dfb56b0df33fda92eb49a5a2c7205621b57a91fd88d7a2cb686d5734a94d |
| SSE_STAR_2025_NOTICE | RETRIEVED | https://star.sse.com.cn/services/listingwithsse/home/policy/supervise/c/c_20250516_10779158.shtml | 457eb1c1807148036f7d9a46085d021112c5f9416bc72ad269d692b3af48e341 |
| SSE_MAIN_2024_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/10785259/files/50bc3d26529d44679cc3e8c94118db0e.docx | 258f8d39ef06ff4c0881a071cb7b36458116f40329fd2d9e42776553868926d7 |
| SSE_MAIN_2024_NOTICE_ATTACH2 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/10785259/files/50bc3d26529d44679cc3e8c94118db0e.docx | 258f8d39ef06ff4c0881a071cb7b36458116f40329fd2d9e42776553868926d7 |
| SSE_STAR_2024_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/10777832/files/c713b7393e1f412aa9efc115ee375d13.docx | f02e8423e467102e30b419f4e01a42600fa315557eca0ac059740114fe849f15 |
| SSE_STAR_2024_NOTICE_ATTACH2 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/10777832/files/c713b7393e1f412aa9efc115ee375d13.docx | f02e8423e467102e30b419f4e01a42600fa315557eca0ac059740114fe849f15 |
| SZSE_MAIN_2024_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/lawrules/rule/repeal/rules/W020240430677262007864.pdf | df0ea12983df0d577fd1f0154d3bf1a1aea2378b0631d6ef2d4c04b0a96bfeea |
| SZSE_CHINEXT_2024_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/lawrules/rule/repeal/rules/W020240528406128801584.pdf | d48d2e9b49b4eb120c506f000992746ea4a5d63d44c1e9943e575ca2ed944967 |
| SSE_MAIN_2020_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/10118801/files/0452199551694160a697fb7488f1cc98.doc | e8a9237324737132ee480c9affb3550cbe503d75305572fd64526021892363b4 |
| SZSE_MAIN_2020_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/general/W020210517344592054829.pdf | 6490507562d641abc5507703a408d99fb2f80d1e0cd479b03c7f7940b133da60 |
| SZSE_MAIN_2012_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/company/W020180328446393013021.pdf | 8540e16b594e2da0268b8c73e3575793fa2c9338e7c7a2dcef434a109b3ea01e |
| SZSE_MAIN_2008_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/company/W020180328445647147870.doc | e65d09b0e81b7e552716e3604f2b07489344945492b1d26ba6f138003bff1e1e |
| SZSE_MAIN_2008_NOTICE_ATTACH2 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/company/W020180328445647151784.doc | ac979a2a89bcf1695924e23e19e6469bdeacf0af169cda01d7d354fe29f80d73 |
| SSE_MAIN_2014_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/10058474/files/3489d480be804a9f8066a1578f4822d7.doc | dcd1ef89dd0d84ea46059794087f7e045a84e1ab986b4446623d189240910440 |
| SSE_MAIN_2018_APR_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/10058222/files/382c60c930c242e38eaeb9c799e196a3.doc | 565baa8b4474722ed8917639d2e9349f12090bf1bed1faf874f5bb44e2a2fe4b |
| SZSE_MAIN_2022_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/general/W020220107650074883940.pdf | 58a95b853b53d4d0e05f0ab1186c4d46d50f8ec4386db3e0d208e3e6a0c53fcd |
| SSE_MAIN_2023_FEB_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/10649677/files/7f696733eae0457fba143d631c01ef61.doc | ae9dee4644a9472121b8d7d07a0f0cae5e4983d1bee5c4ecd24ebe478e67adc8 |
| SZSE_MAIN_2023_AUG_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/W020230804625303228188.pdf | b858a84f8897bb25087062c0f464ee2bf7e47da4d3cfa54b4f0d13cec8274d1d |
| SZSE_CHINEXT_2023_AUG_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/lawrules/rule/repeal/rules/W020230804626744334095.pdf | 1a03a997868792035198a529215fdb57cd1e291e9ca6663a584b2adca834d233 |
| SZSE_CHINEXT_2012_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/company/W020180328446357660655.pdf | 1097c83326b1ce06638f1539cfe656c029ee64289fe797f590c44b03544edd14 |
| SSE_MAIN_2025_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/services/listingwithsse/home/policy/supervise/c/10779023/files/4c610ded661a4f0aa5d29ccf614a0d82.docx | 0cae84ac62ac72c1f935f0ba3c4b23a3f763d444f5706dfb59c37765576e0155 |
| SSE_MAIN_2025_NOTICE_ATTACH2 | RETRIEVED | https://www.sse.com.cn/services/listingwithsse/home/policy/supervise/c/10779023/files/4c610ded661a4f0aa5d29ccf614a0d82.docx | 0cae84ac62ac72c1f935f0ba3c4b23a3f763d444f5706dfb59c37765576e0155 |
| SSE_STAR_2025_NOTICE_ATTACH1 | RETRIEVED | https://star.sse.com.cn/services/listingwithsse/home/policy/supervise/c/10779158/files/e4f92f2712f74c6d8e2cc3c8218a8bbd.docx | f83070466d8c022de7b4e5343afede898b5932942336b4c1ab2e391f1683fadd |
| SSE_STAR_2025_NOTICE_ATTACH2 | RETRIEVED | https://star.sse.com.cn/services/listingwithsse/home/policy/supervise/c/10779158/files/e4f92f2712f74c6d8e2cc3c8218a8bbd.docx | f83070466d8c022de7b4e5343afede898b5932942336b4c1ab2e391f1683fadd |
| SZSE_CHINEXT_2020_DEC_NOTICE | RETRIEVED | https://www.szse.cn/disclosure/notice/t20201231_584051.html | 4152986395e95c8557c781c223f312a38fbbaef1fab6fa46aa60e2e991628864 |
| SZSE_CHINEXT_2020_JUNE_NOTICE | RETRIEVED | https://docs.static.szse.cn/www/lawrules/rule/repeal/announcement/W020210521598995459850.pdf | 9a15f8b18acccae2df233f6ded7a393b99cd1d34f0eefff59cf55c9324d90dc3 |
| SSE_STAR_2019_APR_NOTICE | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/c_20210601_5478483.shtml | df246fa7034be8cbe30dca55296f9663796359061d404017554181369bef774e |
| SSE_MAIN_2023_AUG_NOTICE | FAILED | https://www.sse.com.cn/lawandrules/sselawsrules/stocks/mainipo/c/c_20230804_5724639.shtml |  |
| SSE_STAR_2023_AUG_NOTICE | FAILED | https://www.sse.com.cn/lawandrules/sselawsrules/stocks/staripo/c/c_20230804_5724640.shtml |  |
| SSE_MAIN_2008_EXPLANATION | RETRIEVED | https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988237.shtml | e11eda8ee4fc985080bb6a6e3557ea029d1021f8fea6c5f7ea8f1fac8460851e |
| SZSE_CHINEXT_2020_DEC_NOTICE_ATTACH1 | RETRIEVED | http://docs.static.szse.cn/www/disclosure/notice/W020201231716234749980.pdf | 7305bef1359b7b19d9f4fc81c4dbd2346570dbbc7bc205b25172b01665129b4b |
| SSE_STAR_2019_APR_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules/repeal/rules/c/10118919/files/d08a7af2862f4109a30a25215caac95c.doc | 3c26c6d3f8bee3c605dba5d690c7987e37ebcb1ba70346d80ec9a4aab6fe4aeb |
| SSE_MAIN_2018_NOV_NOTICE | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20181116_10784930.shtml | 4c8dcfc6c6de7ce8ee6cc35b470ae073717ba11bd01b2c050ee81ce3deda52d6 |
| SZSE_MAIN_2025_NOTICE | FAILED | https://www.szse.cn/www/lawrules/rule/allrules/bussiness/t20250425_613256.html |  |
| SZSE_MAIN_2025_RULE | FAILED | https://docs.static.szse.cn/www/lawrules/rule/stock/supervision/mb/W020250425730780478873.pdf |  |
| SSE_MAIN_2018_NOV_NOTICE_ATTACH1 | RETRIEVED | https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/10784930/files/4baf30c32e314bb4848fb1cc3743c78d.doc | 71abc0caeaf8b00ecdaddca073ff3e1254869cf59a551296b0247030d342feec |
