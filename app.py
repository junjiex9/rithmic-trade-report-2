import streamlit as st
import pandas as pd
import numpy as np
import io, os
import plotly.express as px
from datetime import datetime
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
cache_days = st.sidebar.number_input('缓存天数（天）', min_value=1, value=st.session_state['cache_days'])
max_snapshots = st.sidebar.number_input('保留快照份数', min_value=1, value=st.session_state['max_snapshots'])
lookback_days = st.sidebar.slider('回撤回溯期 (天)', 1, 60, value=st.session_state['lookback_days'])
st.session_state.update({'cache_days': cache_days, 'max_snapshots': max_snapshots, 'lookback_days': lookback_days})
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
    dfs = []
    for f in files:
        content = f.getvalue().decode('utf-8', errors='ignore').splitlines()
        idx = next((i for i,l in enumerate(content) if 'Completed Orders' in l), None)
        if idx is None: continue
        header = content[idx+1].replace('"','').split(',')
        body = '\n'.join(content[idx+2:])
        df = pd.read_csv(io.StringIO(body), names=header)
        df['上传文件'] = f.name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Status']=='Filled'][[
        'Account','Buy/Sell','Symbol','Avg Fill Price','Qty To Fill',
        'Update Time (CST)','Commission Fill Rate','Closed Profit/Loss','上传文件'
    ]]
    df.columns = ['账户','方向','品种','价格','数量','时间','手续费','盈亏','上传文件']
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['方向'] = df['方向'].map({'B':'Buy','S':'Sell'})
    for c in ['价格','数量','手续费','盈亏']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['时间','方向']).sort_values('时间').reset_index(drop=True)

df = load_and_clean(uploaded)

# ============ 快照管理 ============
def manage_snapshots(df):
    SNAP_DIR = 'snapshots'
    os.makedirs(SNAP_DIR, exist_ok=True)
    snap = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(os.path.join(SNAP_DIR, snap), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    for old in snaps[:-max_snapshots]: os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，快照：{snap}")
manage_snapshots(df)

# ============ 风险警示 ============
today = datetime.now().date()
trades_today = df[df['时间'].dt.date == today]
if not trades_today.empty and trades_today['盈亏'].min() <= -abs(max_loss):
    st.sidebar.warning(f"{len(trades_today[trades_today['盈亏']<=-abs(max_loss)])} 笔亏损超阈值！")
if len(trades_today) > max_trades:
    st.sidebar.warning(f"今日交易 {len(trades_today)} 次，超阈值！")

# ============ 衍生字段 ============
df['累计盈亏'] = df['盈亏'].cumsum()
df['日期'] = df['时间'].dt.date
df['小时'] = df['时间'].dt.hour

# ============ 统计计算 ============
def compute_stats(data, lookback):
    # calculate core metrics
    if '时间' in data.columns and not data.empty:
        period = max((data['时间'].max() - data['时间'].min()).days, 1)
    else:
        period = 1
    pnl = data['盈亏'] if '盈亏' in data.columns else pd.Series(dtype=float)
    csum = pnl.cumsum()
    sharpe = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() else np.nan
    winrate = (pnl>0).mean() if not pnl.empty else np.nan
    profit_factor = (pnl[pnl>0].sum()/(-pnl[pnl<0].sum())) if (pnl<0).any() else np.nan
    ann = pnl.sum()/period*252 if period else np.nan
    downside = pnl[pnl<0].std() if (pnl<0).any() else np.nan
    var95 = -pnl.quantile(0.05) if not pnl.empty else np.nan
    cvar95 = -pnl[pnl<=pnl.quantile(0.05)].mean() if not pnl.empty else np.nan
    mdd_day = (csum - csum.cummax()).min() if not csum.empty else np.nan
    mdd_look = (csum - csum.rolling(window=lookback, min_periods=1).max()).min() if not csum.empty else np.nan
    return [sharpe, winrate, profit_factor, ann, downside, var95, cvar95, mdd_day, mdd_look]

labels = ['夏普比率','胜率','盈亏比','年化收益率','下行风险','VaR95','CVaR95','最大回撤(当日)','最大回撤(30天)']
today_vals = compute_stats(trades_today, lookback_days)
hist_vals = compute_stats(df, lookback_days)

# ============ UI布局 ============
tabs = st.tabs(['报告视图','数据导出','⚙️ 设置'])

# 1. 报告视图
with tabs[0]:
    dt, ht = st.tabs(['📌 当日统计指标','📌 历史统计指标'])
    # 当日统计指标
    with dt:
        st.subheader('📈 累计盈亏趋势')
        fig1 = px.line(trades_today, x='时间', y='累计盈亏')
        fig1.update_yaxes(tickformat=',.0f')
        st.plotly_chart(fig1, use_container_width=True)
        st.subheader('📊 日/小时盈亏')
        fig2 = px.bar(trades_today.groupby('日期')['盈亏'].sum().reset_index(), x='日期', y='盈亏')
        fig3 = px.bar(trades_today.groupby('小时')['盈亏'].mean().reset_index(), x='小时', y='盈亏')
        for fig in [fig2, fig3]: fig.update_yaxes(tickformat=',.0f'); st.plotly_chart(fig, use_container_width=True)
        st.subheader('⏳ 持仓时长分布（分钟）')
        st.plotly_chart(px.box(trades_today.sort_values(['账户','品种','时间']).assign(持仓时长=lambda d: d.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60), x='账户', y='持仓时长'), use_container_width=True)
        st.subheader('🎲 Monte Carlo 模拟')
        sims = [np.random.choice(trades_today['盈亏'], len(trades_today), replace=True).cumsum()[-1] for _ in range(500)]
        fig4 = px.histogram(sims, nbins=40); fig4.update_yaxes(tickformat=',.0f'); st.plotly_chart(fig4, use_container_width=True)
        if market_file:
            st.subheader('🕳️ 滑点与成交率分析')
            mp = pd.read_csv(market_file); mp['Time']=pd.to_datetime(mp['Time'],errors='coerce')
            merged = trades_today.merge(mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'}), left_on=['品种','时间'], right_on=['品种','Time'], how='left')
            merged['滑点']=merged['价格']-merged['市场价格']
            fig5 = px.histogram(merged, x='滑点'); fig5.update_yaxes(tickformat=',.0f'); st.plotly_chart(fig5, use_container_width=True)
        if sent_file:
            st.subheader('📣 社交舆情热力图')
            ds = pd.read_csv(sent_file); ds['Date']=pd.to_datetime(ds['Date'],errors='coerce').dt.date
            heat = ds.pivot_table(values='SentimentScore', index='Symbol', columns='Date', aggfunc='mean')
            st.plotly_chart(px.imshow(heat, aspect='auto'), use_container_width=True)
    # 历史统计指标
    with ht:
        st.subheader('📈 历史累计盈亏趋势')
        figh = px.line(df, x='时间', y='累计盈亏'); figh.update_yaxes(tickformat=',.0f')
        st.plotly_chart(figh, use_container_width=True)
        st.subheader('📌 核心统计指标')
        cols = st.columns(5)
        for i,(lbl,val) in enumerate(zip(labels, hist_vals)):
            cols[i%5].metric(lbl, f"{val:.2f}")

# 2. 数据导出
with tabs[1]:
    st.subheader('📤 数据导出')
    ce, cp = st.columns(2)
    with ce:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All Trades', index=False)
            pd.DataFrame({'指标':labels,'当日':today_vals,'历史':hist_vals}).to_excel(writer, sheet_name='Stats', index=False)
        st.download_button('📥 下载 Excel 报告', buf.getvalue(), file_name='report.xlsx')
    with cp:
        pdf=FPDF('P','mm','A4'); pdf.set_auto_page_break(True,15); pdf.alias_nb_pages()
        pdf.add_page(); pdf.set_font('Arial','B',16); pdf.cell(0,10,'Automated Trading Analysis Report',ln=1,align='C')
        tmpf='tmp.pdf'; pdf.output(tmpf); data=open(tmpf,'rb').read(); os.remove(tmpf)
        st.download_button('📄 下载 PDF 报告', data, file_name='report.pdf')

# 3. 设置
with tabs[2]:
    st.subheader('⚙️ 设置')
    st.write('在侧边栏调整“缓存天数”、“保留快照份数”、“回撤回溯期”后，刷新应用即可生效。')
