```python
# app.py
# Streamlit 交易分析报告生成器（完整版，含详细图表与导出功能）

import streamlit as st
import pandas as pd
import numpy as np
import io, os
import plotly.express as px
from datetime import datetime

# PDF 导出依赖
try:
    from fpdf import FPDF
    pdf_available = True
except ModuleNotFoundError:
    pdf_available = False

# ============ 页面配置 ============
st.set_page_config(page_title="📈 交易分析报告生成器", layout="wide")

# ============ 多语言支持 ============
LANG = {'中文':'📈 交易分析报告生成器','English':'📈 Trading Report Generator'}
lang = st.sidebar.selectbox('语言 / Language', list(LANG.keys()))
st.title(LANG[lang])

# ============ 侧边栏设置 ============
st.sidebar.header('🔧 设置')
uploaded = st.sidebar.file_uploader('上传 Rithmic/ATAS CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('上传市场快照 CSV', type='csv')
sent_file = st.sidebar.file_uploader('上传舆情 CSV', type='csv')
max_snapshots = st.sidebar.number_input('保留快照数', min_value=1, value=10)
SNAP_DIR = 'snapshots'
os.makedirs(SNAP_DIR, exist_ok=True)

# ============ 数据加载与清洗 ============
@st.cache_data
def load_data(files):
    import io
    rec = []
    for f in files:
        lines = f.getvalue().decode('utf-8', errors='ignore').splitlines()
        idx = next((i for i, l in enumerate(lines) if 'Completed Orders' in l), None)
        if idx is None:
            continue
        header = lines[idx+1].replace('"','').split(',')
        df = pd.read_csv(io.StringIO("\n".join(lines[idx+2:])), names=header)
        df['Source'] = f.name
        rec.append(df)
    df = pd.concat(rec, ignore_index=True)
    df = df[df['Status']=='Filled']
    df = df[['Account','Buy/Sell','Symbol','Avg Fill Price','Qty To Fill','Update Time (CST)','Commission Fill Rate','Closed Profit/Loss','Source']]
    df.columns = ['Account','Direction','Symbol','Price','Qty','Time','Fee','PnL','Source']
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    for c in ['Price','Qty','Fee','PnL']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
    return df

# ============ 主流程 ============
if uploaded:
    df = load_data(uploaded)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 快照管理
    snap = f'snapshot_{now}.csv'
    df.to_csv(os.path.join(SNAP_DIR, snap), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    for old in snaps[:-max_snapshots]:
        os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f'Loaded {len(df)} trades. Snapshot: {snap}')

    # 计算指标
    df['Cumulative'] = df['PnL'].cumsum()
    df['Date'], df['Hour'] = df['Time'].dt.date, df['Time'].dt.hour
    days = max((df['Time'].max() - df['Time'].min()).days, 1)
    total_pnl = df['PnL'].sum()
    ann_return = total_pnl / days * 252
    downside_dev = df[df['PnL']<0]['PnL'].std()
    var95 = -df['PnL'].quantile(0.05)
    cvar95 = -df[df['PnL']<=df['PnL'].quantile(0.05)]['PnL'].mean()
    sharpe = df['PnL'].mean()/df['PnL'].std()*np.sqrt(252) if df['PnL'].std() else 0
    win_rate = (df['PnL']>0).mean()
    profit_factor = df[df['PnL']>0]['PnL'].mean()/(-df[df['PnL']<0]['PnL'].mean())
    max_dd = (df['Cumulative']-df['Cumulative'].cummax()).min()

    # 滑点分析
    if market_file:
        mp = pd.read_csv(market_file)
        mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
        df = df.merge(mp, on=['Symbol','Time'], how='left')
        df['Slippage'] = df['Price'] - df['MarketPrice']
    else:
        df['Slippage'] = np.nan

    # 持仓时长
    df_sorted = df.copy()
    df_sorted['HoldTime'] = df_sorted.groupby(['Account','Symbol'])['Time'].diff().dt.total_seconds()/60

    # Monte Carlo 模拟
    rets = df['PnL'].values; sims, n = 500, len(rets)
    mc_vals = [np.random.choice(rets, n, replace=True).cumsum()[-1] for _ in range(sims)]

    # 舆情热力图
    if sent_file:
        sent = pd.read_csv(sent_file)
        sent['Date'] = pd.to_datetime(sent['Date'], errors='coerce').dt.date
        heat = sent.pivot_table(index='Symbol', columns='Date', values='SentimentScore', aggfunc='mean')
        fig_heat = px.imshow(heat, aspect='auto', title='Sentiment Heatmap')
        fig_heat.write_image('sent_heat.png')

    # 标签页布局
    tabs = st.tabs(['Overview','Charts','Export'])

    # Overview
    with tabs[0]:
        st.subheader('📌 核心统计指标')
        metrics = {
            'Total PnL': total_pnl,
            'Sharpe Ratio': sharpe,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Annual Return': ann_return,
            'Downside Dev': downside_dev,
            'VaR95': var95,
            'CVaR95': cvar95,
            'Max Drawdown': max_dd
        }
        for k,v in metrics.items(): st.metric(k, f'{v:.2f}' if isinstance(v,(int,float)) else v)

    # Charts
    with tabs[1]:
        st.subheader('📈 累计盈亏趋势')
        st.plotly_chart(px.line(df, x='Time', y='Cumulative'), use_container_width=True)
        st.subheader('📊 日/小时盈亏')
        st.plotly_chart(px.bar(df.groupby('Date')['PnL'].sum().reset_index(), x='Date', y='PnL'), use_container_width=True)
        st.plotly_chart(px.bar(df.groupby('Hour')['PnL'].mean().reset_index(), x='Hour', y='PnL'), use_container_width=True)
        st.subheader('⏳ 持仓时长分布（分钟）')
        st.plotly_chart(px.box(df_sorted, x='Account', y='HoldTime'), use_container_width=True)
        st.subheader('🎲 Monte Carlo 模拟')
        st.plotly_chart(px.histogram(mc_vals, nbins=40), use_container_width=True)
        if not df['Slippage'].isna().all():
            st.subheader('🕳️ 滑点分析')
            st.plotly_chart(px.histogram(df, x='Slippage'), use_container_width=True)
        if os.path.exists('sent_heat.png'):
            st.subheader('📣 社交舆情热力图')
            st.image('sent_heat.png', use_column_width=True)

    # Export
    with tabs[2]:
        # Excel 导出
        st.subheader('📥 导出Excel报告')
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as ew:
            df.to_excel(ew, sheet_name='Trades', index=False)
            df.groupby('Date')['PnL'].sum().to_excel(ew, sheet_name='DailyPL')
            df.groupby('Hour')['PnL'].mean().to_excel(ew, sheet_name='HourlyPL')
            df.groupby('Account')['PnL'].agg(['sum','count','mean','std']).to_excel(ew, sheet_name='AccountStats')
            df.groupby('Symbol')['PnL'].agg(['sum','count','mean','std']).to_excel(ew, sheet_name='SymbolStats')
            df.assign(Month=df['Time'].dt.to_period('M')).groupby('Month')['PnL'].sum().to_frame().to_excel(ew, sheet_name='MonthlyPL')
            df_sorted[['Account','Symbol','HoldTime']].to_excel(ew, sheet_name='Durations', index=False)
            pd.DataFrame(metrics, index=[0]).T.reset_index(names=['Metric','Value']).to_excel(ew, sheet_name='Summary', index=False)
        st.download_button('Download Excel Report', data=excel_buffer.getvalue(), file_name=f'Report_{now}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # PDF 导出
        st.subheader('📄 导出PDF报告')
        if pdf_available and st.button('Download PDF Report'):
            pdf = FPDF()
            pdf.set_auto_page_break(True, margin=15)
            # 使用之前示例的详细 PDF 写入逻辑
            from app_pdf import write_full_pdf  # 假设提取为模块
            write_full_pdf(pdf, df, df_sorted, mc_vals, metrics, now)
            buf = io.BytesIO()
            pdf.output(buf)
            st.download_button('Download PDF Report', buf.getvalue(), file_name=f'Report_{now}.pdf', mime='application/pdf')
        elif not pdf_available:
            st.info('PDF 导出未启用，请安装 fpdf2 & kaleido')
else:
    st.info('👆 上传 CSV 文件以开始分析')
```
