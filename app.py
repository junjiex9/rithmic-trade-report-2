import streamlit as st
import pandas as pd
import numpy as np
import io, os
import plotly.express as px
from datetime import datetime
from fpdf import FPDF

# ============ 页面配置 ============
st.set_page_config(
    page_title="📈 自动化交易分析报告生成器",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# ============ 多语言支持 ============
LANG = {'中文': '📈 自动化交易分析报告生成器', 'English': '📈 Automated Trading Report Generator'}
lang = st.sidebar.selectbox('语言 / Language', list(LANG.keys()))
st.title(LANG[lang])

# ============ 侧边栏上传 ============
st.sidebar.header('📁 上传与设置')
uploaded = st.sidebar.file_uploader('上传交易 CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('市场快照 CSV', type='csv')
sent_file = st.sidebar.file_uploader('舆情数据 CSV', type='csv')

if not uploaded:
    st.sidebar.info('请上传交易CSV以开始。')
    st.stop()

# ============ 默认参数 ============
DEFAULT_CACHE_DAYS = 1
DEFAULT_MAX_SNAPSHOTS = 10
DEFAULT_LOOKBACK = 30

# ============ 数据加载 ============
@st.cache_data(show_spinner=False)
def load_and_clean(files):
    def extract_orders(f):
        content = f.getvalue().decode('utf-8', errors='ignore').splitlines()
        idx = next((i for i,l in enumerate(content) if 'Completed Orders' in l), None)
        if idx is None:
            return pd.DataFrame()
        header = content[idx+1].replace('"','').split(',')
        body = "\n".join(content[idx+2:])
        df = pd.read_csv(io.StringIO(body), names=header)
        df['上传文件'] = f.name
        return df
    dfs = [extract_orders(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Status']=='Filled']
    df = df[[
        'Account','Buy/Sell','Symbol','Avg Fill Price','Qty To Fill',
        'Update Time (CST)','Commission Fill Rate','Closed Profit/Loss','上传文件'
    ]]
    df.columns = ['账户','方向','品种','价格','数量','时间','手续费','盈亏','上传文件']
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['方向'] = df['方向'].map({'B':'Buy','S':'Sell'})
    for c in ['价格','数量','手续费','盈亏']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['时间','方向']).sort_values('时间').reset_index(drop=True)

# load data
df = load_and_clean(uploaded)

# derive fields
df['累计盈亏'] = df['盈亏'].cumsum()
df['日期'] = df['时间'].dt.date
df['小时'] = df['时间'].dt.hour

# metric preparation
def compute_metrics(lookback):
    period_days = max((df['时间'].max() - df['时间'].min()).days, 1)
    total_pl = df['盈亏'].sum()
    sharpe = df['盈亏'].mean() / df['盈亏'].std() * np.sqrt(252)
    winrate = (df['盈亏'] > 0).mean()
    profit_factor = df[df['盈亏']>0]['盈亏'].mean() / -df[df['盈亏']<0]['盈亏'].mean()
    mdd = (df['累计盈亏'] - df['累计盈亏'].cummax()).min()
    calmar = total_pl/period_days*252/abs(mdd) if mdd<0 else np.nan
    roll_max = df['累计盈亏'].rolling(window=lookback).max()
    recent_dd = (df['累计盈亏'] - roll_max).min()
    return sharpe, winrate, profit_factor, mdd, calmar, recent_dd

# UI布局
tabs = st.tabs(['报告视图','数据导出','⚙️ 设置'])

# 1. 报告视图
with tabs[0]:
    st.subheader('📈 累计盈亏趋势')
    fig1 = px.line(df, x='时间', y='累计盈亏')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader('📊 日盈亏 & 小时盈亏')
    fig2 = px.bar(df.groupby('日期')['盈亏'].sum().reset_index(), x='日期', y='盈亏', title='日盈亏')
    fig3 = px.bar(df.groupby('小时')['盈亏'].mean().reset_index(), x='小时', y='盈亏', title='小时盈亏')
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader('⏳ 持仓时长分布（分钟）')
    sorted_df = df.sort_values(['账户','品种','时间'])
    sorted_df['持仓时长'] = sorted_df.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60
    fig4 = px.box(sorted_df, x='账户', y='持仓时长', title='按账户')
    fig5 = px.box(sorted_df, x='品种', y='持仓时长', title='按品种')
    st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader('🎲 Monte Carlo 模拟')
    sims = [np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1] for _ in range(500)]
    fig6 = px.histogram(sims, nbins=40, title='Monte Carlo 分布')
    st.plotly_chart(fig6, use_container_width=True)

    if market_file:
        st.subheader('🕳️ 滑点分析')
        mp = pd.read_csv(market_file)
        mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
        mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'}, inplace=True)
        merged = df.merge(mp, left_on=['品种','时间'], right_on=['品种','Time'], how='left')
        merged['滑点'] = merged['价格'] - merged['市场价格']
        fig7 = px.histogram(merged, x='滑点', title='滑点分布')
        st.plotly_chart(fig7, use_container_width=True)

    if sent_file:
        st.subheader('📣 舆情热力图')
        ds = pd.read_csv(sent_file)
        ds['Date'] = pd.to_datetime(ds['Date'], errors='coerce').dt.date
        heat = ds.pivot_table(values='SentimentScore', index='Symbol', columns='Date')
        fig8 = px.imshow(heat, aspect='auto', title='舆情热力图')
        st.plotly_chart(fig8, use_container_width=True)

    st.subheader('📌 核心指标')
    sharpe, winrate, pf, mdd, calmar, recent_dd = compute_metrics(st.session_state.get('lookback_days', DEFAULT_LOOKBACK))
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric('夏普率', f"{sharpe:.2f}")
    c2.metric('胜率', f"{winrate:.2%}")
    c3.metric('盈亏比', f"{pf:.2f}")
    c4.metric('最大回撤', f"{mdd:.2f}")
    c5.metric('回撤(天)', f"{recent_dd:.2f}")
    c6.metric('Calmar', f"{calmar:.2f}")

# 2. 数据导出
with tabs[1]:
    st.subheader('📤 数据导出')
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Trades', index=False)
        daily = df.groupby('日期')['盈亏'].sum().reset_index()
        daily.to_excel(writer, sheet_name='Daily PnL', index=False)
        hourly = df.groupby('小时')['盈亏'].mean().reset_index()
        hourly.to_excel(writer, sheet_name='Hourly PnL', index=False)
        sorted_df[['账户','品种','持仓时长']].to_excel(writer, sheet_name='Holding Time', index=False)
        pd.DataFrame({'Simulation PnL': sims}).to_excel(writer, sheet_name='Monte Carlo', index=False)
        metrics_df = pd.DataFrame({
            '指标':['总交易次数','总盈亏','夏普率','胜率','盈亏比','最大回撤','Calmar'],
            '数值':[len(df), df['盈亏'].sum(), sharpe, winrate, pf, mdd, calmar]
        })
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    st.download_button('下载 Excel (.xlsx)', excel_buffer.getvalue(), 'detailed_report.xlsx')
    
    # PDF 导出
    pdf = FPDF('P','mm','A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    # 封面
    pdf.add_page()
    pdf.set_font('Arial','B',20)
    pdf.cell(0,60,'',ln=1)
    pdf.cell(0,10,'Automated Trading Report',ln=1,align='C')
    pdf.set_font('Arial','',12)
    pdf.cell(0,10,f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',ln=1,align='C')
    # Core Metrics
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Core Metrics',ln=1)
    pdf.set_font('Arial','',12)
    for _, row in metrics_df.iterrows():
        pdf.cell(50,8,str(row['指标']))
        pdf.cell(0,8,str(row['数值']),ln=1)
    # Monte Carlo图
    pdf.add_page()
    mc_fig = px.histogram(sims, nbins=40)
    mc_img = mc_fig.to_image(format='png', width=600, height=300)
    pdf.image(io.BytesIO(mc_img), x=15, y=pdf.get_y()+5, w=180)
    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button('下载 PDF 报告', pdf_bytes, 'detailed_report.pdf')

# 3. 设置
with tabs[1]:
    # ... (data export code unchanged)
    pass

# 3. 设置
with tabs[2]:
    st.subheader('⚙️ 设置')
    # 二级菜单控件
    cache_days = st.number_input('缓存天数（天）', min_value=1, value=st.session_state.get('cache_days', DEFAULT_CACHE_DAYS))
    max_snapshots = st.number_input('保留快照份数', min_value=1, value=st.session_state.get('max_snapshots', DEFAULT_MAX_SNAPSHOTS))
    lookback_days = st.slider('回撤回溯期 (天)', 1, 60, value=st.session_state.get('lookback_days', DEFAULT_LOOKBACK))
    # 保存设置
    st.session_state['cache_days'] = cache_days
    st.session_state['max_snapshots'] = max_snapshots
    st.session_state['lookback_days'] = lookback_days
    st.write('👍 参数已更新，将在下一次运行时生效。')
