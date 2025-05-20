import streamlit as st
import pandas as pd
import numpy as np
import io, os
import plotly.express as px
from datetime import datetime, date
from fpdf import FPDF

# ============ 页面配置 ============
st.set_page_config(
    page_title="📈 Rithmic 自动化交易分析报告生成器",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ============ 多语言支持 ============
LANG = {'中文': '📈 Rithmic 自动化交易分析报告生成器', 'English': '📈 Automated Trading Report Generator'}
lang = st.sidebar.selectbox('语言 / Language', list(LANG.keys()))
st.title(LANG[lang])

# ============ 默认参数 ============
DEFAULT_CACHE_DAYS, DEFAULT_MAX_SNAPSHOTS, DEFAULT_LOOKBACK = 1, 10, 30
for key, default in [('cache_days', DEFAULT_CACHE_DAYS), ('max_snapshots', DEFAULT_MAX_SNAPSHOTS), ('lookback_days', DEFAULT_LOOKBACK)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============ 侧边栏：上传与设置 ============
st.sidebar.header('📁 上传与设置')
uploaded = st.sidebar.file_uploader('上传交易 CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('市场快照 CSV', type='csv')
sent_file = st.sidebar.file_uploader('舆情数据 CSV', type='csv')

# 缓存、快照、回撤设置
cache_days = st.sidebar.number_input('缓存天数（天）', min_value=1, value=st.session_state['cache_days'])
max_snapshots = st.sidebar.number_input('保留快照份数', min_value=1, value=st.session_state['max_snapshots'])
lookback_days = st.sidebar.slider('回撤回溯期 (天)', 1, 60, value=st.session_state['lookback_days'])
st.session_state.update({'cache_days': cache_days, 'max_snapshots': max_snapshots, 'lookback_days': lookback_days})

# 历史日期范围选择
min_date, max_date = df_min:=date.today().replace(day=1), date.today()
st.sidebar.write('📅 选择历史分析日期范围')
hist_range = st.sidebar.date_input('历史日期范围', [min_date, max_date], key='hist_range')
hist_start, hist_end = (hist_range if isinstance(hist_range, list) else [min_date, max_date])

if not uploaded:
    st.sidebar.info('请上传交易CSV以开始。')
    st.stop()

# ============ 风险阈值预警 ============
st.sidebar.header('⚠️ 风险阈值预警')
max_loss = st.sidebar.number_input('单笔最大亏损', value=100.0)
max_trades = st.sidebar.number_input('日内最大交易次数', value=50)

# ============ 数据加载 ============
@st.cache_data(show_spinner=False, max_entries=10, ttl=3600*24*cache_days)
def load_and_clean(files):
    records = []
    for f in files:
        content = f.getvalue().decode('utf-8', errors='ignore').splitlines()
        idx = next((i for i,l in enumerate(content) if 'Completed Orders' in l), None)
        if idx is None: continue
        hdr = content[idx+1].replace('"','').split(',')
        body = '\n'.join(content[idx+2:])
        df0 = pd.read_csv(io.StringIO(body), names=hdr)
        df0['上传文件'] = f.name
        records.append(df0)
    df = pd.concat(records, ignore_index=True)
    df = df[df['Status']=='Filled'][[
        'Account','Buy/Sell','Symbol','Avg Fill Price','Qty Filled',
        'Update Time (CST)','Commission Fill Rate','P&L','上传文件'
    ]]
    df.columns = ['账户','方向','品种','价格','数量','时间','手续费','盈亏','上传文件']
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['方向'] = df['方向'].map({'B':'Buy','S':'Sell'})
    for col in ['价格','数量','手续费','盈亏']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['时间','方向']).sort_values('时间').reset_index(drop=True)

df = load_and_clean(uploaded)

# 衍生字段
df['累计盈亏'] = df['盈亏'].cumsum()
df['日期'] = df['时间'].dt.date
df['小时'] = df['时间'].dt.hour

# 快照管理
def manage_snapshots(df):
    path = 'snapshots'
    os.makedirs(path, exist_ok=True)
    snap = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(os.path.join(path, snap), index=False)
    snaps = sorted(os.listdir(path))
    for old in snaps[:-max_snapshots]: os.remove(os.path.join(path, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，快照：{snap}")
manage_snapshots(df)

# 当日数据与风险提示
today = datetime.now().date()
trades_today = df[df['日期']==today]
if not trades_today.empty:
    if trades_today['盈亏'].min() <= -abs(max_loss):
        st.sidebar.warning(f"{len(trades_today[trades_today['盈亏']<=-abs(max_loss)])} 笔亏损超阈值！")
    if len(trades_today) > max_trades:
        st.sidebar.warning(f"今日交易 {len(trades_today)} 次，超阈值！")

# 统计函数
def compute_stats(data, lookback):
    if data.empty:
        return [np.nan]*10
    pnl = data['盈亏']
    csum = pnl.cumsum()
    days = max((data['时间'].max()-data['时间'].min()).days,1)
    sharpe = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() else np.nan
    win = (pnl>0).mean()
    pf = pnl[pnl>0].sum()/(-pnl[pnl<0].sum()) if (pnl<0).any() else np.nan
    ann = pnl.sum()/days*252
    down = pnl[pnl<0].std()
    var95 = -pnl.quantile(0.05)
    cvar95 = -pnl[pnl<=pnl.quantile(0.05)].mean()
    mdd_d = (csum-csum.cummax()).min()
    mdd_l = (csum-csum.rolling(window=lookback,min_periods=1).max()).min()
    hist = df['盈亏'].cumsum(); mdd_h = (hist-hist.cummax()).min()
    return [sharpe,win,pf,ann,down,var95,cvar95,mdd_d,mdd_l,mdd_h]

labels = ['夏普比率','胜率','盈亏比','年化收益率','下行风险','VaR95','CVaR95','最大回撤(当日)','最大回撤(回溯)','最大回撤(历史)']
today_stats = compute_stats(trades_today, lookback_days)
hist_df = df[(df['日期']>=hist_start)&(df['日期']<=hist_end)]
hist_stats = compute_stats(hist_df, lookback_days)

# ============ UI布局 ============
tabs_main = st.tabs(['报告视图','数据导出','⚙️ 设置'])

# 报告视图
with tabs_main[0]:
    tab_today, tab_hist = st.tabs(['📌 当日统计指标','📌 历史统计指标'])
    with tab_today:
        st.subheader('当日交易概览')
        st.dataframe(trades_today)
        for title, fig in [
            ('📈 累计盈亏趋势', px.line(trades_today,x='时间',y='累计盈亏')),
            ('📊 日/小时盈亏', px.bar(trades_today.groupby('日期')['盈亏'].sum().reset_index(),x='日期',y='盈亏')),
            ('⏳ 持仓时长分布（分钟）', px.box(trades_today.assign(持仓时长=trades_today.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60),x='账户',y='持仓时长')),
            ('🎲 Monte Carlo 模拟', px.histogram([np.random.choice(trades_today['盈亏'],len(trades_today)) for _ in range(500)],nbins=40))
        ]:
            fig.update_yaxes(tickformat=',.0f'); st.subheader(                                                                                      title); st.plotly_chart(fig,use_container_width=True)
        if market_file:
            st.subheader('🕳️ 滑点与成交率分析')
            mp = pd.read_csv(market_file); mp['Time']=pd.to_datetime(mp['Time']); mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'},inplace=True)
            merged = trades_today.merge(mp,left_on=['品种','时间'],right_on=['品种','Time'],how='left'); merged['滑点']=merged['价格']-merged['市场价格']
            fig_sp=px.histogram(merged,x='滑点'); fig_sp.update_yaxes(tickformat=',.0f'); st.plotly_chart(fig_sp,use_container_width=True)
        if sent_file:
            st.subheader('📣 社交舆情热力图')
            ss=pd.read_csv(sent_file); ss['Date']=pd.to_datetime(ss['Date']).dt.date
            fig_ht=px.imshow(ss.pivot_table(values='SentimentScore',index='Symbol',columns='Date'),aspect='auto'); st.plotly_chart(fig_ht,use_container_width=True)
        st.subheader('📌 核心统计指标')
        cols=st.columns(4)
        for i,(lbl,val) in enumerate(zip(labels,today_stats)):
            cols[i%4].metric(lbl,f"{val:.2f}")
    with tab_hist:
        st.subheader('历史交易概览')
        st.dataframe(hist_df)
        for title, fig in [
            ('📈 累计盈亏趋势 (历史)',px.line(hist_df,x='时间',y='累计盈亏')),
            ('📊 日/小时盈亏 (历史)',px.bar(hist_df.groupby('日期')['盈亏'].sum().reset_index(),x='日期',y='盈亏'))
        ]:
            fig.update_yaxes(tickformat=',.0f'); st.subheader(title); st.plotly_chart(fig,use_container_width=True)
        st.subheader('📌 核心统计指标')
        cols=st.columns(4)
        for i,(lbl,val) in enumerate(zip(labels,hist_stats)):
            cols[i%4].metric(lbl,f"{val:.2f}")

# 数据导出
with tabs_main[1]:
    st.subheader('📤 数据导出')
    ce, cp = st.columns(2)
    with ce:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, 'AllTrades', index=False)
            pd.DataFrame({'指标':labels,'当日':today_stats,'历史':hist_stats}).to_excel(writer,'Stats',index=False)
        st.download_button('📥 下载 Excel', buf.getvalue(), 'report.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    with cp:
        pdf = FPDF('P','mm','A4'); pdf.set_auto_page_break(True,15); pdf.alias_nb_pages(); pdf.add_page()
        pdf.set_font('Arial','B',16); pdf.cell(0,10,'Automated Trading Report',ln=1,align='C')
        tmp_pdf='tmp.pdf'; pdf.output(tmp_pdf); data=open(tmp_pdf,'rb').read(); os.remove(tmp_pdf)
        st.download_button('📄 下载 PDF', data, 'report.pdf',mime='application/pdf')

# 设置
with tabs_main[2]:
    st.subheader('⚙️ 设置')
    st.write('请在侧边栏调整“缓存天数”、“保留快照份数”、“回撤回溯期”与“历史日期范围”，
             然后刷新应用以生效。')
