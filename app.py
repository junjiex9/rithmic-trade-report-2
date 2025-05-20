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

# ============ 侧边栏上传与设置 ============
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
        if idx is None:
            continue
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

# 管理快照
def manage_snapshots(df):
    SNAP_DIR = 'snapshots'
    os.makedirs(SNAP_DIR, exist_ok=True)
    snap = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(os.path.join(SNAP_DIR, snap), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    for old in snaps[:-max_snapshots]: os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，快照：{snap}")
manage_snapshots(df)

# 风险警示
today = datetime.now().date()
trades_today = df[df['时间'].dt.date==today]
if not trades_today.empty and trades_today['盈亏'].min() <= -abs(max_loss):
    st.sidebar.warning(f"{len(trades_today[trades_today['盈亏']<=-abs(max_loss)])} 笔亏损超阈值！")
if len(trades_today)>max_trades:
    st.sidebar.warning(f"今日交易{len(trades_today)}次, 超阈值！")

# 衍生字段
df['累计盈亏'] = df['盈亏'].cumsum()
df['日期'] = df['时间'].dt.date
df['小时'] = df['时间'].dt.hour

# 统计计算 --- 核心统计指标
def compute_stats(data, lookback):
    period_days = max((data['时间'].max() - data['时间'].min()).days, 1)
    pnl = data['盈亏']; csum = pnl.cumsum()
    sharpe = pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() else np.nan
    winrate = (pnl>0).mean()
    profit_factor = pnl[pnl>0].sum()/(-pnl[pnl<0].sum()) if pnl.min()<0 else np.nan
    ann_return = pnl.sum()/period_days*252
    downside_dev = pnl[pnl<0].std()
    var95 = -pnl.quantile(0.05)
    cvar95 = -pnl[pnl<=pnl.quantile(0.05)].mean()
    daily_mdd = (csum - csum.cummax()).min()
    lookback_mdd = (csum - csum.rolling(window=lookback).max()).min()
    hist_mdd = (csum - csum.cummax()).min()
    return [
        sharpe, winrate, profit_factor, ann_return,
        downside_dev, var95, cvar95, daily_mdd,
        lookback_mdd, hist_mdd
    ]

labels = [
    '夏普比率','胜率','盈亏比','年化收益率','下行风险',
    'VaR95','CVaR95','最大回撤 (当日)','最大回撤 (30天)','最大回撤（历史）'
]
today_vals = compute_stats(trades_today, lookback_days)
hist_vals = compute_stats(df, lookback_days)

# UI布局
tabs = st.tabs(['报告视图','数据导出','⚙️ 设置'])

# 1. 报告视图
with tabs[0]:
    st.subheader('📌 当日统计指标')
    cols = st.columns(5)
    for i,(lbl,val) in enumerate(zip(labels,today_vals)):
        disp = f"{val:.2f}" if isinstance(val,float) else str(val)
        cols[i%5].metric(lbl,disp)

    st.subheader('📈 累计盈亏趋势')
    st.plotly_chart(px.line(df,x='时间',y='累计盈亏').update_yaxes(tickformat=',.0f'),use_container_width=True)

    st.subheader('📊 日/小时盈亏')
    figd = px.bar(df.groupby('日期')['盈亏'].sum().reset_index(),x='日期',y='盈亏')
    figh = px.bar(df.groupby('小时')['盈亏'].mean().reset_index(),x='小时',y='盈亏')
    figd.update_yaxes(tickformat=',.0f'); figh.update_yaxes(tickformat=',.0f')
    st.plotly_chart(figd,use_container_width=True)
    st.plotly_chart(figh,use_container_width=True)

    st.subheader('⏳ 持仓时长分布（分钟）')
    sd = df.sort_values(['账户','品种','时间'])
    sd['持仓时长'] = sd.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60
    st.plotly_chart(px.box(sd,x='账户',y='持仓时长'),use_container_width=True)

    st.subheader('🎲 Monte Carlo 模拟')
    sims = [np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1] for _ in range(500)]
    hist = px.histogram(sims,nbins=40).update_yaxes(tickformat=',.0f')
    st.plotly_chart(hist,use_container_width=True)

    if market_file:
        st.subheader('🕳️ 滑点与成交率分析')
        mp = pd.read_csv(market_file)
        mp['Time'] = pd.to_datetime(mp['Time'],errors='coerce')
        merged = df.merge(mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'}), left_on=['品种','时间'], right_on=['品种','Time'], how='left')
        merged['滑点'] = merged['价格']-merged['市场价格']
        fig = px.histogram(merged,x='滑点').update_yaxes(tickformat=',.0f')
        st.plotly_chart(fig,use_container_width=True)

    if sent_file:
        st.subheader('📣 社交舆情热力图')
        ds = pd.read_csv(sent_file)
        ds['Date'] = pd.to_datetime(ds['Date'],errors='coerce').dt.date
        heat = ds.pivot_table(values='SentimentScore',index='Symbol',columns='Date')
        st.plotly_chart(px.imshow(heat,aspect='auto'),use_container_width=True)

    st.subheader('📌 历史统计指标')
    cols = st.columns(5)
    for i,(lbl,val) in enumerate(zip(labels,hist_vals)):
        disp = f"{val:.2f}" if isinstance(val,float) else str(val)
        cols[i%5].metric(lbl,disp)

# 2. 数据导出
with tabs[1]:
    st.subheader('📤 数据导出')
    col_excel, col_pdf = st.columns(2)

    with col_excel:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Trades', index=False)
            pd.DataFrame({
                '指标': labels,
                '当日': today_vals,
                '历史': hist_vals
            }).to_excel(writer, sheet_name='Stats', index=False)
        st.download_button(
            '下载 Excel (.xlsx)', excel_buf.getvalue(),
            file_name='report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    with col_pdf:
        pdf = FPDF('P','mm','A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()
        pdf.set_font('Helvetica','',12)
        # 封面
        pdf.add_page()
        pdf.set_font('Helvetica','B',16)
        pdf.cell(0,10,'Automated Trading Analysis Report',ln=1,align='C')
        pdf.set_font('Helvetica','',10)
        pdf.cell(0,10,f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',ln=1,align='C')
        # 核心统计指标
        pdf.add_page()
        pdf.set_font('Helvetica','B',14)
        pdf.cell(0,10,'📌 核心统计指标',ln=1)
        pdf.set_font('Helvetica','',12)
        for lbl, val in zip(labels, hist_vals):
            pdf.cell(60,8,lbl)
            pdf.cell(0,8,f"{val:.2f}" if isinstance(val,float) else str(val),ln=1)
        # Monte Carlo 图
        pdf.add_page()
        pdf.set_font('Helvetica','B',14)
        pdf.cell(0,10,'Monte Carlo Distribution',ln=1)
        mc_img = px.histogram(sims, nbins=40).to_image(format='png', width=600, height=300)
        tmp_img = 'temp_mc.png'
        with open(tmp_img,'wb') as f_img: f_img.write(mc_img)
        pdf.image(tmp_img, x=15, y=pdf.get_y()+5, w=180)
        os.remove(tmp_img)
        tmp_pdf = 'temp_report.pdf'
        pdf.output(tmp_pdf)
        with open(tmp_pdf,'rb') as f_pdf: pdf_bytes = f_pdf.read()
        st.download_button('下载 PDF 报告', pdf_bytes, file_name='report.pdf', mime='application/pdf')

# 3. 设置
with tabs[2]:
    st.subheader('⚙️ 设置')
    st.write('在侧边栏调整 缓存天数、快照保留、回撤回溯期 后刷新以生效。')
