# app.py
# Streamlit 交易分析报告生成器（修正版，动态图表与导出功能）

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
st.set_page_config(page_title="📈 Rithmic 交易分析报告生成器", layout="wide")

# ============ 多语言支持 ============
LANG = {'中文':'📈 Rithmic 交易分析报告生成器','English':'📈 Trading Report Generator'}
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
    rec = []
    for f in files:
        text = f.getvalue().decode('utf-8', errors='ignore')
        lines = text.splitlines()
        idx = next((i for i, l in enumerate(lines) if 'Completed Orders' in l), None)
        if idx is None:
            continue
        header = [h.strip().strip('"') for h in lines[idx+1].split(',')]
        df0 = pd.read_csv(io.StringIO("\n".join(lines[idx+2:])), names=header)
        df0['Source'] = f.name
        rec.append(df0)
    if not rec:
        return pd.DataFrame()
    df = pd.concat(rec, ignore_index=True)
    if 'Status' in df.columns:
        df = df[df['Status']=='Filled']
    # 动态映射列名称
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if 'account' in lc:
            rename_map[col] = 'Account'
        elif 'buy' in lc and 'sell' in lc:
            rename_map[col] = 'Direction'
        elif 'direction' in lc:
            rename_map[col] = 'Direction'
        elif 'symbol' in lc:
            rename_map[col] = 'Symbol'
        elif 'fill price' in lc or 'avg fill' in lc:
            rename_map[col] = 'Price'
        elif 'qty' in lc:
            rename_map[col] = 'Qty'
        elif 'update time' in lc or lc=='time':
            rename_map[col] = 'Time'
        elif 'commission' in lc:
            rename_map[col] = 'Fee'
        elif 'profit' in lc:
            rename_map[col] = 'PnL'
        elif col=='Source':
            rename_map[col] = 'Source'
    df = df.rename(columns=rename_map)
    # 只保留必要列
    need = ['Account','Direction','Symbol','Price','Qty','Time','Fee','PnL','Source']
    df = df[[c for c in need if c in df.columns]]
    # 时间列转换
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    # 数值列转换
    for c in ['Price','Qty','Fee','PnL']:
        if c in df.columns:
            col_data = df[c]
            if not isinstance(col_data, pd.Series):
                col_data = pd.Series(col_data)
            df[c] = pd.to_numeric(col_data, errors='coerce')
    df = df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
    return df

# ============ 主流程 ============
if uploaded:
    df = load_data(uploaded)
    if df.empty:
        st.error('未能解析到成交记录，请检查上传文件格式。')
    else:
        now = datetime.now().strftime('%Y%m%d_%H%M%S')
        snap = f'snapshot_{len(uploaded)}files_{now}.csv'
        df.to_csv(os.path.join(SNAP_DIR, snap), index=False)
        snaps = sorted(os.listdir(SNAP_DIR))
        for old in snaps[:-max_snapshots]:
            os.remove(os.path.join(SNAP_DIR, old))
        st.sidebar.success(f'Loaded {len(df)} trades. Snapshot: {snap}')

        # 核心指标计算
        df['Cumulative'] = df['PnL'].cumsum()
        df['Date'] = df['Time'].dt.date
        df['Hour'] = df['Time'].dt.hour
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
            if 'MarketPrice' in mp.columns:
                mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
                df = df.merge(mp[['Symbol','Time','MarketPrice']], on=['Symbol','Time'], how='left')
                df['Slippage'] = df['Price'] - df['MarketPrice']
            else:
                df['Slippage'] = np.nan
        else:
            df['Slippage'] = np.nan

        # 持仓时长分布
        df_sorted = df.copy()
        df_sorted['HoldTime'] = df_sorted.groupby(['Account','Symbol'])['Time'].diff().dt.total_seconds()/60

        # Monte Carlo 模拟
        sims, n = 500, len(df)
        mc_vals = [np.random.choice(df['PnL'], n, replace=True).cumsum()[-1] for _ in range(sims)]

        # 舆情热力图
        heat_png = 'sent_heat.png'
        if sent_file:
            sent = pd.read_csv(sent_file)
            if set(['SentimentScore','Symbol','Date']).issubset(sent.columns):
                sent['Date'] = pd.to_datetime(sent['Date'], errors='coerce').dt.date
                heat = sent.pivot_table(index='Symbol', columns='Date', values='SentimentScore', aggfunc='mean')
                fig_heat = px.imshow(heat, aspect='auto', title='Sentiment Heatmap')
                fig_heat.write_image(heat_png)

        tabs = st.tabs(['Overview','Charts','Export'])
        with tabs[0]:
            st.subheader('📌 核心统计指标')
            metrics = dict(TotalPnL=total_pnl, Sharpe=sharpe, WinRate=win_rate, ProfitFactor=profit_factor, AnnualReturn=ann_return, DownsideDev=downside_dev, VaR95=var95, CVaR95=cvar95, MaxDD=max_dd)
            for k,v in metrics.items(): st.metric(k, f'{v:.2f}')
        with tabs[1]:
            st.subheader('📈 累计盈亏趋势')
            st.plotly_chart(px.line(df, x='Time', y='Cumulative', title='Cumulative PnL'), use_container_width=True)
            st.subheader('📊 日/小时盈亏')
            st.plotly_chart(px.bar(df.groupby('Date')['PnL'].sum().reset_index(), x='Date', y='PnL', title='Daily PnL'), use_container_width=True)
            st.plotly_chart(px.bar(df.groupby('Hour')['PnL'].mean().reset_index(), x='Hour', y='PnL', title='Hourly PnL'), use_container_width=True)
            st.subheader('⏳ 持仓时长分布')
            st.plotly_chart(px.box(df_sorted.dropna(subset=['HoldTime']), x='Account', y='HoldTime', title='Hold Time Distribution'), use_container_width=True)
            st.subheader('🎲 Monte Carlo 模拟')
            st.plotly_chart(px.histogram(mc_vals, nbins=40, title='Monte Carlo Distribution'), use_container_width=True)
            if df['Slippage'].notna().any():
                st.subheader('🕳️ 滑点分析')
                st.plotly_chart(px.histogram(df, x='Slippage', title='Slippage Distribution'), use_container_width=True)
            if os.path.exists(heat_png):
                st.subheader('📣 舆情热力图')
                st.image(heat_png, use_column_width=True)
        with tabs[2]:
            st.subheader('📥 导出Excel报告')
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as ew:
                df.to_excel(ew, sheet_name='Trades', index=False)
                df.groupby('Date')['PnL'].sum().to_excel(ew, sheet_name='DailyPL')
                df.groupby('Hour')['PnL'].mean().to_excel(ew, sheet_name='HourlyPL')
                df.groupby('Account')['PnL'].agg(['sum','count','mean','std']).to_excel(ew, sheet_name='AccountStats')
                df.groupby('Symbol')['PnL'].agg(['sum','count','mean','std']).to_excel(ew, sheet_name='SymbolStats')
                df.assign(Month=df['Time'].dt.to_period('M')).groupby('Month')['PnL'].sum().to_frame().to_excel(ew, sheet_name='MonthlyPL')
                df_sorted[['Account','Symbol','HoldTime']].dropna().to_excel(ew, sheet_name='Durations', index=False)
                pd.DataFrame(metrics, index=[0]).T.reset_index(names=['Metric','Value']).to_excel(ew, sheet_name='Summary', index=False)
            st.download_button('Download Excel Report', buf.getvalue(), file_name=f'Report_{now}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            
            st.subheader('📄 导出PDF报告（详细表格与图像）')
            if pdf_available and st.button('Download PDF Report'):
                pdf = FPDF()
                pdf.set_auto_page_break(True, margin=15)
                pdf.add_page()
                pdf.set_font('Arial','B',16)
                pdf.cell(0,10,'交易分析报告',ln=True,align='C')
                pdf.ln(5)
                pdf.set_font('Arial','',12)
                pdf.cell(0,8,f'生成时间: {now}', ln=True)
                pdf.cell(0,8,f'Total PnL: {total_pnl:.2f}   Sharpe: {sharpe:.2f}', ln=True)
                pdf.cell(0,8,f'Total PnL: {total`
}]}
