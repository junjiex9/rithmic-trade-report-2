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
title_list = [
    '📈 累计盈亏趋势','📊 日盈亏','📊 小时盈亏',
    '⏳ 持仓时长分布','🎲 Monte Carlo 模拟',
    '🕳️ 滑点分析','📣 舆情热力图','📌 核心指标'
]
with tabs[0]:
    st.subheader('📈 累计盈亏趋势')
    st.plotly_chart(px.line(df, x='时间', y='累计盈亏'), use_container_width=True)
    st.subheader('📊 日盈亏')
    st.plotly_chart(px.bar(df.groupby('日期')['盈亏'].sum().reset_index(), x='日期', y='盈亏'), use_container_width=True)
    st.subheader('📊 小时盈亏')
    st.plotly_chart(px.bar(df.groupby('小时')['盈亏'].mean().reset_index(), x='小时', y='盈亏'), use_container_width=True)
    st.subheader('⏳ 持仓时长分布')
    df_sorted = df.sort_values(['账户','品种','时间'])
    df_sorted['持仓时长'] = df_sorted.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60
    st.plotly_chart(px.box(df_sorted, x='账户', y='持仓时长'), use_container_width=True)
    st.plotly_chart(px.box(df_sorted, x='品种', y='持仓时长'), use_container_width=True)
    st.subheader('🎲 Monte Carlo 模拟')
    sims = [np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1] for _ in range(500)]
    st.plotly_chart(px.histogram(sims, nbins=50), use_container_width=True)
    if market_file:
        st.subheader('🕳️ 滑点分析')
        mp = pd.read_csv(market_file)
        mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
        mp.rename(columns={'MarketPrice':'市场价','Symbol':'品种'}, inplace=True)
        dfm = df.merge(mp, left_on=['品种','时间'], right_on=['品种','Time'], how='left')
        dfm['滑点'] = dfm['价格'] - dfm['市场价']
        st.plotly_chart(px.histogram(dfm, x='滑点'), use_container_width=True)
    if sent_file:
        st.subheader('📣 舆情热力图')
        ds = pd.read_csv(sent_file)
        ds['Date'] = pd.to_datetime(ds['Date'], errors='coerce').dt.date
        heat = ds.pivot_table(values='SentimentScore', index='Symbol', columns='Date', aggfunc='mean')
        st.plotly_chart(px.imshow(heat, aspect='auto'), use_container_width=True)
    st.subheader('📌 核心指标')
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    sharpe, winrate, pf, mdd, calmar, recent_dd = compute_metrics(st.session_state.get('lookback_days', DEFAULT_LOOKBACK))
    c1.metric('夏普率', f"{sharpe:.2f}")
    c2.metric('胜率', f"{winrate:.2%}")
    c3.metric('盈亏比', f"{pf:.2f}")
    c4.metric('最大回撤', f"{mdd:.2f}")
    c5.metric('回撤（天）', f"{recent_dd:.2f}")
    c6.metric('Calmar', f"{calmar:.2f}")

# 2. 数据导出
with tabs[1]:
    st.subheader('📤 数据导出')
    # 导出 Excel，多表格
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # 原始交易数据
        df.to_excel(writer, sheet_name='Trades', index=False)
        # 核心指标
        sharpe, winrate, pf, mdd, calmar, recent_dd = compute_metrics(
            st.session_state.get('lookback_days', DEFAULT_LOOKBACK)
        )
        metrics_df = pd.DataFrame({
            '指标': ['总交易次数','总盈亏','夏普率','胜率','盈亏比','最大回撤','Calmar'],
            '数值': [len(df), df['盈亏'].sum(), sharpe, winrate, pf, mdd, calmar]
        })
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        # 日/小时 盈亏
        daily = df.groupby('日期')['盈亏'].sum().reset_index()
        daily.to_excel(writer, sheet_name='Daily PnL', index=False)
        hourly = df.groupby('小时')['盈亏'].mean().reset_index()
        hourly.to_excel(writer, sheet_name='Hourly PnL', index=False)
        # 持仓时长
        df_sorted = df.sort_values(['账户','品种','时间'])
        df_sorted['持仓时长（分）'] = (
            df_sorted.groupby(['账户','品种'])['时间']
            .diff().dt.total_seconds()/60
        )
        df_sorted[['账户','品种','持仓时长（分）']].to_excel(
            writer, sheet_name='Holding Time', index=False
        )
        # Monte Carlo 模拟结果
        sims = [
            np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1]
            for _ in range(500)
        ]
        pd.DataFrame({'Simulation PnL': sims}).to_excel(
            writer, sheet_name='Monte Carlo', index=False
        )
    st.download_button(
        '下载 Excel (.xlsx)', excel_buffer.getvalue(), file_name='detailed_report.xlsx'
    )

    # 导出 PDF，包含指标表格和图表
    # 确保 sims 已定义
    pdf = FPDF('P','mm','A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    # 封面
    pdf.add_page()
    pdf.set_font('Arial','B',20)
    pdf.cell(0,60,'', ln=1)
    pdf.cell(0,10,'交易分析报告', ln=1, align='C')
    pdf.set_font('Arial','',12)
    pdf.cell(
        0,10, f'生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}', ln=1, align='C'
    )
    # 核心指标
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'核心统计指标', ln=1)
    pdf.set_font('Arial','',12)
    for _, row in metrics_df.iterrows():
        pdf.cell(50,8,str(row['指标']))
        pdf.cell(0,8,str(row['数值']), ln=1)
    # Monte Carlo 图
    pdf.add_page()
    mc_fig = px.histogram(sims, nbins=40)
    mc_img = mc_fig.to_image(format='png', width=600, height=300)
    pdf.image(io.BytesIO(mc_img), x=15, y=pdf.get_y()+5, w=180)
    # 导出按钮
    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button(
        '下载 PDF 报告', pdf_bytes, file_name='detailed_report.pdf'
    )

# 3. 设置[1]:
    st.subheader('📤 数据导出')
    # 导出 Excel，多表格
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # 原始交易数据
        df.to_excel(writer, sheet_name='Trades', index=False)
        # 核心指标
        sharpe, winrate, pf, mdd, calmar, recent_dd = compute_metrics(st.session_state.get('lookback_days', DEFAULT_LOOKBACK))
        metrics_df = pd.DataFrame({
            '指标': ['总交易次数','总盈亏','夏普率','胜率','盈亏比','最大回撤','Calmar'],
            '数值': [
                len(df), df['盈亏'].sum(), sharpe, winrate, pf, mdd, calmar
            ]
        })
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        # 日/小时 盈亏
        daily = df.groupby('日期')['盈亏'].sum().reset_index()
        daily.to_excel(writer, sheet_name='Daily PnL', index=False)
        hourly = df.groupby('小时')['盈亏'].mean().reset_index()
        hourly.to_excel(writer, sheet_name='Hourly PnL', index=False)
        # 持仓时长
        df_sorted = df.sort_values(['账户','品种','时间'])
        df_sorted['持仓时长（分）'] = df_sorted.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60
        holding = df_sorted[['账户','品种','持仓时长（分）']]
        holding.to_excel(writer, sheet_name='Holding Time', index=False)
        # Monte Carlo 模拟结果
        sims = [np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1] for _ in range(500)]
        pd.DataFrame({'Simulation PnL': sims}).to_excel(writer, sheet_name='Monte Carlo', index=False)
    st.download_button('下载 Excel (.xlsx)', excel_buffer.getvalue(), file_name='detailed_report.xlsx')

        # 导出 PDF，包含指标表格
    # 重新计算指标表格
    sharpe, winrate, pf, mdd, calmar, recent_dd = compute_metrics(st.session_state.get('lookback_days', DEFAULT_LOOKBACK))
    metrics_df = pd.DataFrame({
        '指标': ['总交易次数','总盈亏','夏普率','胜率','盈亏比','最大回撤','Calmar'],
        '数值': [len(df), df['盈亏'].sum(), sharpe, winrate, pf, mdd, calmar]
    })
    pdf = FPDF('P','mm','A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.alias_nb_pages()
    # 使用 UTF-8 支持：删除表情，使用默认字体（仅支持 Latin; 中文可能需要额外字体）
    # 封面
    pdf.add_page()
    pdf.set_font('Arial','B',20)
    pdf.cell(0,60,'',ln=1)
    pdf.cell(0,10,'交易分析报告',ln=1,align='C')
    pdf.set_font('Arial','',12)
    pdf.cell(0,10,f'生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}',ln=1,align='C')
    # 核心指标
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'核心统计指标',ln=1)
    pdf.set_font('Arial','',12)
    for _, row in metrics_df.iterrows():
        pdf.cell(50,8,str(row['指标']))
        pdf.cell(0,8,str(row['数值']),ln=1)
    # 日度盈亏
    daily = df.groupby('日期')['盈亏'].sum().reset_index()
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'日度盈亏（前10条）',ln=1)
    pdf.set_font('Arial','',10)
    for _, r in daily.head(10).iterrows():
        pdf.cell(40,6,str(r['日期']))
        pdf.cell(0,6,f"{r['盈亏']:.2f}",ln=1)
    # Monte Carlo 分布
    pdf.add_page()
    pdf.set_font('Arial','B',16)
    pdf.cell(0,10,'Monte Carlo 模拟分布',ln=1)
    mc_fig = px.histogram(sims, nbins=40)
    mc_img = mc_fig.to_image(format='png', width=600, height=300)
    pdf.image(io.BytesIO(mc_img), x=15, y=pdf.get_y()+5, w=180)
    # 下载按钮
    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button('下载 PDF 报告', pdf_bytes, file_name='detailed_report.pdf')
