from .classify import operative,iso,risk_match
def test_classify_by_operative_article():
    assert operative('未在期限内披露年度报告，触及第10.3.10条第一款第四项。')['category']=='C1'
    assert operative('第9.5.2条第一款第六项')['category']=='C4'
    assert operative('公司被立案调查，最终结果未知') is None
def test_two_trading_triggers_one_statutory_category():
    s='每日收盘价均低于1元、每日收盘总市值均低于5亿元。第9.2.1条第(一)项、第(五)项'
    assert risk_match('C2','price',s) and risk_match('C2','cap',s)
    assert operative(s)['category']=='C2' and '(五)项' in operative(s)['article']
def test_calendar_date():
    assert iso('2024年2月29日')=='2024-02-29'
