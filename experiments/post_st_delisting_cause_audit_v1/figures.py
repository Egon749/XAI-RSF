"""Publication figures from released aggregates only. No company data or thresholds."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common import D,pd,np
C=['#246776','#bc6633','#697589'];out=D/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def figure(title,sub,figsize=(8.4,4.6)):
    fig,ax=plt.subplots(figsize=figsize);fig.subplots_adjust(left=.15,right=.97,bottom=.25,top=.78)
    fig.text(.04,.95,title,fontsize=15,weight='bold',color='#18364b');fig.text(.04,.875,sub,fontsize=9,color='#42566a');return fig,ax
def save(fig,name,foot):
    fig.text(.04,.095,foot,fontsize=8,color='#42566a');fig.text(.04,.041,'C039 (17, 503, 22), 96 trees  |  34 direct exchange decisions + 4 issuer filings quoting decisions  |  Exploratory',fontsize=8,color='#42566a')
    for ext in ('pdf','png'):fig.savefig(out/(name+'.'+ext),dpi=200,facecolor='white')
    plt.close(fig)
def table(ax,labels,rows):
    ax.axis('off');t=ax.table(cellText=rows,colLabels=labels,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.75)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor('#1d5364' if r==0 else ['#f3f6f7','#e5edf0'][r%2])
        if r==0:cell.get_text().set_color('white')
def main():
    df=pd.read_csv(D/'CAUSE_COUNTS_BY_YEAR.csv');n=df.query("scope=='all_observed'").pivot(index='entry_year',columns='category',values='n')
    fig,ax=figure('Grounds of observed final delisting','Distinct firms by entry cohort and operative legal category. Counts include observed late exits.')
    table(ax,['Entry year','Financial','Trading','Normative','Major violation','Total'],[[str(y),*n.loc[y,['C1','C2','C3','C4']].astype(int),int(n.loc[y].sum())] for y in n.index])
    save(fig,'01_observed_by_entry_year','38 confirmed exits. Voluntary/other, unresolved multi-category and unknown: zero. The 2025 cohort has incomplete H24 follow-up.')
    fig,ax=figure('How the outcome window changes the case mix','Number of observed events within calendar horizons. Each row counts firms once.')
    dates=[('H12','12 months','2023–25'),('supported_H24','24 months','2023–24')]
    rows=[]
    for scope,h,y in dates:
        cc=df[df.scope==scope].groupby('category').n.sum();rows.append([h,y,*cc[['C1','C2','C3','C4']].astype(int),sum(cc)])
    table(ax,['Window','Entry years','Financial','Trading','Normative','Major violation','Total'],rows)
    save(fig,'02_horizon_counts','Some of the same events appear in both rows. The 2025 H24 window has no formal performance estimate.')
    a=pd.read_csv(D/'CAUSE_SIGNAL_COMPARISON.csv').query("scope=='pooled'");idx=['four_capabilities','G5','G6','G7']
    fig,ax=figure('Which signals raise or lower overall delisting risk?','C039 mean signed contributions to the overall delisting CIF, in percentage points.')
    for i,(h,cat,label) in enumerate([(12,'C2_pure','12m trading, n=21 (2023–25)'),(24,'C2_pure','24m trading, n=23 (2023–24)'),(24,'C1','24m financial, n=8 (2023–24)')]):
        ar=a[(a.months==h)&(a.group==cat)].set_index('signal');ax.scatter(ar.loc[idx].signed_mean,np.arange(4)+(i-1)*.15,color=C[i],marker=['o','s','D'][i],label=label,s=42)
    ax.axvline(0,color='#9baab0',lw=1);ax.set_yticks(range(4),['Four capabilities','Cash / financing','Market','Audit']);ax.set_xlabel('Mean contribution (percentage points)');ax.legend(fontsize=8,loc='lower right')
    save(fig,'03_economic_signals','Pure trading excludes dual price-and-cap triggers. Sparse categories (1–4) have no means. Reference is the training marginal product.')
    d=pd.read_csv(D/'TRADING_VS_NONTRADING.csv').query("statistic=='signed_mean'").set_index('signal')
    fig,ax=figure('Market contrasts depend on the cohort','24 months, trading minus non-trading. Non-trading events in this window were all financial.')
    ids=['G6','alloc_log_price_last','alloc_log_total_mktcap_last'];pos=np.arange(len(ids))
    for i,(field,label) in enumerate([('difference','Pooled 2023–24: 23 vs 8'),('common_year_equal_weight_difference','2023 only: 6 vs 5')]):
        ax.scatter(d.loc[ids,field],pos+(i-.5)*.18,c=C[i],marker=['o','s'][i],s=55,label=label)
    ax.set_yticks(pos,['Market group','Last price','Total market cap']);ax.set_xlabel('Difference in mean signed contribution (percentage points)');ax.axvline(0,lw=1,c='#81979f');ax.legend(fontsize=8)
    save(fig,'04_market_contrast','2024 has three financial events. Common-year comparison uses only years with at least five events in each group.')
    top=pd.read_csv(D/'CAUSE_TOP20_CAPTURE.csv');rows=[]
    for h,_,y in [(12,None,'2023–25'),(24,None,'2023–24')]:
        for name,label in [('C2_pure','Trading, single trigger'),('C1','Financial'),('C4','Major violation')]:
            r=top.query("scope=='pooled' and months==@h and group==@name").iloc[0]
            rows.append([y,f'{h}m',label,r.n,'—' if pd.isna(r.captured) else f'{int(r.captured)} / {r.n}'])
    fig,ax=figure('Events captured in the annual top 20%','Annual quotas are 8, 9 and 11 firms. The cohort is ranked on overall delisting risk.',(8.4,5))
    table(ax,['Cohorts','Window','Realized ground','Events','Captured'],rows)
    save(fig,'05_top20','Dash: fewer than five events or none. One dual trading trigger is outside the 24m pure-trading row. Windows overlap.')
    z=pd.read_csv(D/'SUPPORTED_RAW_PROFILES.csv').query("months==24");fig,ax=figure('Entry price and capitalization in the native risk functions','24m raw main effect averaged within training-defined thirds; 81 evaluation firms, 32 events (2023–24).')
    for i,(v,year,label) in enumerate([(v,y,f'{y}  '+('Price' if v=='log_price_last' else 'Total cap')) for v in ['log_price_last','log_total_mktcap_last'] for y in [2023,2024]]):
        q=z[(z.feature==v)&(z.year==year)];ax.scatter(q.bin+(i-1.5)*.075,q.mean_raw_main_pp,label=label,marker=['o','s','o','s'][i],color=C[i//2],s=47)
    ax.axhline(0,c='#9baab0',lw=1);ax.set_xticks([0,1,2],['Lower third','Middle third','Upper third']);ax.set_ylabel('Overall CIF contribution (pp)');ax.legend(fontsize=8,ncol=2)
    save(fig,'06_price_cap_function','At least 10 training firms, 3 training exits and 5 evaluation firms per displayed cell. One opening snapshot cannot establish a 20-day breach.')
if __name__=='__main__':main()
