# app.py
# Streamlit 交易分析报告生成器（增强版 — 分账户/品种分析、Monte‑Carlo 置信区间、滑点与持仓时长统计，完善 PDF/XLSX 导出）

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import plotly.express as px
from datetime import datetime

# -------- PDF 依赖 --------
try:
    from fpdf import FPDF
    pdf_available = True
except ModuleNotFoundError:
    pdf_available = False

# -------- 页面配置 --------
st.set_page_config(page_title="📈 Rithmic 交易分析报告生成器", layout="wide")

# -------- 多语言 --------
LANG = {'中文': '📈 Rithmic 交易分析报告生成器', 'English': '📈 Trading Report Generator'}
lang = st.sidebar.selectbox('语言 / Language', list(LANG.keys()))
st.title(LANG[lang])

# -------- 侧边栏 --------
st.sidebar.header('🔧 设置')
uploaded = st.sidebar.file_uploader('上传 Rithmic/ATAS CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('上传市场快照 CSV', type='csv')
sent_file = st.sidebar.file_uploader('上传舆情 CSV', type='csv')
max_snapshots = st.sidebar.number_input('保留快照数', min_value=1, value=10, step=1)
SNAP_DIR = 'snapshots'
os.makedirs(SNAP_DIR, exist_ok=True)

# -------- 数据加载 --------
@st.cache_data(show_spinner=False)
def load_data(files):
    records = []
    for f in files:
        text = f.getvalue().decode('utf-8', errors='ignore')
        lines = text.splitlines()
        # 定位 Completed Orders
        idx = next((i for i, l in enumerate(lines) if 'Completed Orders' in l), None)
        if idx is None:
            continue
        header = [h.strip().strip('"') for h in lines[idx+1].split(',')]
        df0 = pd.read_csv(io.StringIO("\n".join(lines[idx+2:])), names=header)
        df0['Source'] = f.name
        records.append(df0)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    # 只保留已成交
    if 'Status' in df.columns:
        df = df[df['Status'] == 'Filled']
    # 动态字段映射
    rename_map = {}
    for col in df.columns:
        lc = col.lower()
        if 'account' in lc:
            rename_map[col] = 'Account'
        elif 'buy' in lc and 'sell' in lc or 'direction' in lc:
            rename_map[col] = 'Direction'
        elif 'symbol' in lc:
            rename_map[col] = 'Symbol'
        elif 'fill price' in lc or 'avg fill' in lc:
            rename_map[col] = 'Price'
        elif 'qty' in lc:
            rename_map[col] = 'Qty'
        elif lc == 'time' or 'update time' in lc:
            rename_map[col] = 'Time'
        elif 'commission' in lc:
            rename_map[col] = 'Fee'
        elif 'profit' in lc:
            rename_map[col] = 'PnL'
        elif col == 'Source':
            rename_map[col] = 'Source'
    df = df.rename(columns=rename_map)
    # 保留必要列
    cols = ['Account','Direction','Symbol','Price','Qty','Time','Fee','PnL','Source']
    df = df[[c for c in cols if c in df.columns]]
    # 类型转换
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    for c in ['Price','Qty','Fee','PnL']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
    return df

# ================= 主流程 =================
if uploaded:
    df = load_data(uploaded)
    if df.empty:
        st.error('未能解析到成交记录，请检查上传的 CSV 格式。')
        st.stop()

    # 快照管理
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    snap_file = f'snapshot_{len(uploaded)}files_{now}.csv'
    df.to_csv(os.path.join(SNAP_DIR, snap_file), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    for old in snaps[:-max_snapshots]:
        try:
            os.remove(os.path.join(SNAP_DIR, old))
        except:
            pass
    st.sidebar.success(f'Loaded {len(df)} trades · 快照: {snap_file}')

    # 派生数据
    df['Date'] = df['Time'].dt.date
    df['Hour'] = df['Time'].dt.hour
    df['Cumulative'] = df['PnL'].cumsum()

    # 核心指标
    pnl = df['PnL']
    days = max((df['Time'].max()-df['Time'].min()).days,1)
    metrics = {
        'Total PnL': pnl.sum(),
        'Sharpe': pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() else 0,
        'Win Rate': (pnl>0).mean(),
        'Profit Factor': pnl[pnl>0].mean()/(-pnl[pnl<0].mean()) if (pnl<0).any() else np.nan,
        'Ann Return': pnl.sum()/days*252,
        'Downside Dev': pnl[pnl<0].std(),
        'VaR95': -pnl.quantile(0.05),
        'CVaR95': -pnl[pnl<=pnl.quantile(0.05)].mean(),
        'Max Drawdown': (df['Cumulative']-df['Cumulative'].cummax()).min()
    }
    acct_stats = df.groupby('Account')['PnL'].agg(['sum','count','mean','std']).reset_index()
    sym_stats = df.groupby('Symbol')['PnL'].agg(['sum','count','mean','std']).reset_index()

    # 滑点
    if market_file:
        mp = pd.read_csv(market_file)
        if {'Symbol','Time','MarketPrice'}.issubset(mp.columns):
            mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
            df = df.merge(mp[['Symbol','Time','MarketPrice']], on=['Symbol','Time'], how='left')
            df['Slippage'] = df['Price']-df['MarketPrice']
        else:
            df['Slippage']=np.nan
    else:
        df['Slippage']=np.nan
    slip_stats = df['Slippage'].dropna().agg(['mean','median', lambda x: x.quantile(0.95),'std']).rename({'<lambda_0>':'95%Quantile'})

    # 持仓时长
    df2 = df.copy()
    df2['HoldTime'] = df2.groupby(['Account','Symbol'])['Time'].diff().dt.total_seconds()/60
    hold_stats = df2['HoldTime'].dropna().agg(['mean','median', lambda x: x.quantile(0.95)]).rename({'<lambda_0>':'95%Quantile'})

    # Monte Carlo
    sims, n = 1000, len(pnl)
    mc_end = [np.random.choice(pnl, n, replace=True).cumsum()[-1] for _ in range(sims)]
    mc_ci = pd.Series(np.percentile(mc_end,[2.5,50,97.5]), index=['2.5%','50%','97.5%'])

    # 舆情热力图
    heat_png='sent_heat.png'
    if sent_file:
        sent=pd.read_csv(sent_file)
        if {'SentimentScore','Symbol','Date'}.issubset(sent.columns):
            sent['Date']=pd.to_datetime(sent['Date'],errors='coerce').dt.date
            heat_df=sent.pivot_table(index='Symbol',columns='Date',values='SentimentScore',aggfunc='mean')
            fig=px.imshow(heat_df,aspect='auto',title='舆情热力图')
            try:
                fig.write_image(heat_png)
            except:
                pass

    # UI 标签页
    tabs = st.tabs(['Overview','Charts','Deep-Dive','Export'])
    with tabs[0]:
        st.subheader('📌 核心统计指标')
        cols = st.columns(3)
        for i,(k,v) in enumerate(metrics.items()): cols[i%3].metric(k,f"{v:.2f}")
        st.divider()
        st.markdown('**分账户统计**')
        st.dataframe(acct_stats,use_container_width=True)
        st.markdown('**分品种统计**')
        st.dataframe(sym_stats,use_container_width=True)

    with tabs[1]:
        st.subheader('📈 累计盈亏趋势')
        st.plotly_chart(px.line(df,x='Time',y='Cumulative',title='累计盈亏'),use_container_width=True)
        st.subheader('📊 日/小时盈亏')
        daily=df.groupby('Date')['PnL'].sum().reset_index()
        hourly=df.groupby('Hour')['PnL'].mean().reset_index()
        st.plotly_chart(px.bar(daily,x='Date',y='PnL',title='每日盈亏'),use_container_width=True)
        st.plotly_chart(px.bar(hourly,x='Hour',y='PnL',title='每小时盈亏'),use_container_width=True)

    with tabs[2]:
        st.subheader('🕳️ 滑点分析')
        if df['Slippage'].notna().any(): st.plotly_chart(px.histogram(df,x='Slippage',title='滑点分布'),use_container_width=True); st.table(slip_stats.to_frame('Value'))
        else: st.info('无滑点数据')
        st.subheader('⏳ 持仓时长')
        if df2['HoldTime'].notna().any(): st.plotly_chart(px.box(df2.dropna(subset=['HoldTime']),x='Symbol',y='HoldTime',title='持仓时长分布'),use_container_width=True); st.table(hold_stats.to_frame('Value'))
        else: st.info('无持仓时长数据')
        st.subheader('🎲 Monte Carlo 置信区间')
        st.table(mc_ci.to_frame('PnL'))
        if os.path.exists(heat_png): st.subheader('📣 舆情热力图'); st.image(heat_png,use_column_width=True)

    with tabs[3]:
        # Excel 导出
        st.subheader('📥 导出 Excel 报告')
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine='xlsxwriter') as ew:
            df.to_excel(ew,sheet_name='Trades',index=False)
            daily.to_excel(ew,sheet_name='DailyPL',index=False)
            hourly.to_excel(ew,sheet_name='HourlyPL',index=False)
            acct_stats.to_excel(ew,sheet_name='AccountStats',index=False)
            sym_stats.to_excel(ew,sheet_name='SymbolStats',index=False)
            df2.dropna(subset=['HoldTime']).to_excel(ew,sheet_name='Durations',index=False)
            slip_stats.to_frame('Value').to_excel(ew,sheet_name='SlippageStats')
            hold_stats.to_frame('Value').to_excel(ew,sheet_name='HoldStats')
            mc_ci.to_frame('PnL').to_excel(ew,sheet_name='MonteCarloCI')
            pd.DataFrame(metrics,index=[0]).T.reset_index(names=['Metric','Value']).to_excel(ew,sheet_name='Summary',index=False)
        st.download_button('Download Excel Report',data=buf.getvalue(),file_name=f'Report_{now}.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # PDF 导出
        st.subheader('📄 导出 PDF 报告')
        if pdf_available and st.button('Download PDF Report'):
            pdf=FPDF()
            pdf.set_auto_page_break(True,margin=15)
            # 封面
            pdf.add_page()
            pdf.set_font('Arial','B',16)
            pdf.cell(0,10,'交易分析报告',ln=True,align='C')
            pdf.ln(5)
            pdf.set_font('Arial','',12)
            pdf.cell(0,8,f'生成时间: {now}',ln=True)
            pdf.cell(0,8,f"Total PnL: {metrics['Total PnL']:.2f}   Sharpe: {metrics['Sharpe']:.2f}",ln=True)
            pdf.ln(5)
            # 核心指标
            df_sum=pd.DataFrame(metrics,index=[0]).T.reset_index(names=['Metric','Value'])
            w=pdf.epw/2
            pdf.set_font('Arial','B',12)
            pdf.cell(w,8,'Metric',border=1); pdf.cell(w,8,'Value',border=1); pdf.ln()
            pdf.set_font('Arial','',10)
            for _,r in df_sum.iterrows(): pdf.cell(w,8,r['Metric'],border=1); pdf.cell(w,8,f"{r['Value']:.2f}",border=1); pdf.ln()
            # 账户&品种统计
            for title,table in [('账户统计',acct_stats),('品种统计',sym_stats)]:
                pdf.add_page(); pdf.set_font('Arial','B',14); pdf.cell(0,8,title,ln=True)
                cols=table.columns.tolist(); w0=pdf.epw/len(cols)
                pdf.set_font('Arial','B',10)
                for c in cols: pdf.cell(w0,6,c,border=1)
                pdf.ln(); pdf.set_font('Arial','',9)
                for _,row in table.iterrows():
                    for c in cols: pdf.cell(w0,6,str(row[c]),border=1)
                    pdf.ln()
            buf_pdf=io.BytesIO(); pdf.output(buf_pdf)
            st.download_button('Download PDF Report',data=buf_pdf.getvalue(),file_name=f'Report_{now}.pdf',mime='application/pdf')
        elif not pdf_available:
            st.info('PDF 导出未启用，请安装 fpdf')
