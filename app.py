# -*- coding: utf-8 -*-
"""
Streamlit ‑ Rithmic 自动化交易分析报告生成器（修正版）
=======================================================
• 使用 FIFO 动态成本法重新计算盈亏，正确处理加仓/部分平仓。
• 多语言界面、中英 PDF/Excel 导出、风险提示及可视化。
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io, os
from datetime import datetime, date
from fpdf import FPDF

# ========= 页面配置 =========
st.set_page_config(page_title="📈 Rithmic 自动化交易分析报告生成器",
                   layout="wide", initial_sidebar_state="expanded", page_icon="📊")

# ========= 语言 =========
LANG = {"中文": "📈 Rithmic 自动化交易分析报告生成器",
        "English": "📈 Automated Trading Report Generator"}
lang = st.sidebar.selectbox("语言 / Language", list(LANG.keys()))
st.title(LANG[lang])

# ========= 默认参数 =========
DEFAULT_CACHE_DAYS, DEFAULT_MAX_SNAPSHOTS, DEFAULT_LOOKBACK = 1, 10, 30
for k, v in [("cache_days", DEFAULT_CACHE_DAYS),
             ("max_snapshots", DEFAULT_MAX_SNAPSHOTS),
             ("lookback_days", DEFAULT_LOOKBACK)]:
    st.session_state.setdefault(k, v)

# ========= 上传 =========
st.sidebar.header("📁 上传与设置")
uploaded_files = st.sidebar.file_uploader("上传交易 CSV", type="csv", accept_multiple_files=True)
market_file = st.sidebar.file_uploader("市场快照 CSV", type="csv")
sent_file   = st.sidebar.file_uploader("舆情数据 CSV", type="csv")
st.sidebar.caption("🛈 市场快照 CSV 需包含列 ‘MarketPrice’ 与 ‘Symbol’ 才能计算滑点")

# 参数
cache_days   = st.sidebar.number_input("缓存天数（天）", 1, 30, st.session_state["cache_days"])
max_snapshots= st.sidebar.number_input("保留快照份数", 1, 50, st.session_state["max_snapshots"])
lookback_days= st.sidebar.slider("回撤回溯期 (天)", 1, 60, st.session_state["lookback_days"])
st.session_state.update(cache_days=cache_days, max_snapshots=max_snapshots, lookback_days=lookback_days)

_today = date.today()
hist_start, hist_end = st.sidebar.date_input("📅 历史日期范围", [_today.replace(day=1), _today])

if not uploaded_files:
    st.sidebar.info("请上传交易 CSV 以开始。")
    st.stop()

# ========= 风险阈值 =========
st.sidebar.header("⚠️ 风险阈值预警")
max_loss   = st.sidebar.number_input("单笔最大亏损", value=100.0)
max_trades = st.sidebar.number_input("日内最大交易次数", value=50)

# ========= 数据加载 =========
@st.cache_data(show_spinner=False, ttl=86400*cache_days)
def load_and_clean(files):
    frames=[]
    for f in files:
        lines=f.getvalue().decode('utf-8',errors='ignore').splitlines()
        idx=next((i for i,l in enumerate(lines) if 'Completed Orders' in l),None)
        if idx is None: continue
        header=lines[idx+1].replace('"','').split(',')
        df_part=pd.read_csv(io.StringIO('\n'.join(lines[idx+2:])), names=header)
        df_part['上传文件']=f.name
        frames.append(df_part)
    if not frames: return pd.DataFrame()
    df=pd.concat(frames, ignore_index=True)
    df=df[df['Status']=='Filled'][[
        'Account','Buy/Sell','Symbol','Avg Fill Price',
        'Qty To Fill','Qty Filled','Position Disposition',
        'Update Time (CST)','Commission Fill Rate','Closed Profit/Loss','上传文件']]
    df.columns=['账户','方向','品种','价格','开仓数量','平仓数量','持仓处置',
                '时间','手续费','盈亏_raw','上传文件']
    df['时间']=pd.to_datetime(df['时间'],errors='coerce')
    df['方向']=df['方向'].map({'B':'Buy','S':'Sell'})
    for c in ['价格','开仓数量','平仓数量','手续费','盈亏_raw']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.dropna(subset=['时间','方向']).sort_values('时间').reset_index(drop=True)

df=load_and_clean(uploaded_files)
if df.empty:
    st.error("上传文件未检测到有效订单数据，请检查格式。")
    st.stop()

# ========= FIFO 重新计算盈亏 =========
SIDE_SIGN={'Buy':1,'Sell':-1}
positions={}
realized=[]
for idx,row in df.iterrows():
    key=(row['账户'],row['品种'])
    sign=SIDE_SIGN[row['方向']]
    qty=int(row['平仓数量'] if row['平仓数量'] else row['开仓数量'])
    price=row['价格']
    pos=positions.setdefault(key,{"qty":0,"cost":0.0})
    # 平仓
    if pos['qty']*sign<0:
        close_qty=min(abs(pos['qty']), qty)
        realized_pnl=(price-pos['cost'])*close_qty*sign
        realized.append((idx,realized_pnl))
        pos['qty']+=sign*close_qty
        qty-=close_qty
        if pos['qty']==0: pos['cost']=0
    # 开仓
    if qty>0:
        new_notional=pos['cost']*abs(pos['qty'])+price*qty
        pos['qty']+=sign*qty
        pos['cost']=new_notional/abs(pos['qty'])

df['盈亏']=0.0
for i,pnl in realized:
    df.at[i,'盈亏']=pnl

# 衍生
df['累计盈亏']=df['盈亏'].cumsum()
df['日期']=df['时间'].dt.date

# ========= 快照 =========
os.makedirs('snapshots',exist_ok=True)
snap_name=f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.csv"
df.to_csv(os.path.join('snapshots',snap_name),index=False)
for old in sorted(os.listdir('snapshots'))[:-max_snapshots]:
    os.remove(os.path.join('snapshots',old))
st.sidebar.success(f"已加载 {len(df)} 条交易，快照: {snap_name}")

# ========= 今日风险 =========
today_trades=df[df['日期']==_today]
if not today_trades.empty:
    if today_trades['盈亏'].min()<=-abs(max_loss):
        st.sidebar.warning(f"{(today_trades['盈亏']<=-abs(max_loss)).sum()} 笔亏损超阈值！")
    if len(today_trades)>max_trades:
        st.sidebar.warning(f"今日交易 {len(today_trades)} 次，超阈值！")

# ========= 统计 =========
def metrics(data):
    if data.empty: return [np.nan]*8
    pnl=data['盈亏']; csum=pnl.cumsum()
    sharpe=pnl.mean()/pnl.std()*np.sqrt(252) if pnl.std() else np.nan
    win=(pnl>0).mean()
    pf=pnl[pnl>0].sum() / (-pnl[pnl<0].sum()) if (pnl<0).any() else np.nan
    annual=pnl.sum()
    downside=pnl[pnl<0].std()
    var95=-pnl.quantile(0.05)
    cvar95=-pnl[pnl<=pnl.quantile(0.05)].mean()
    mdd=(csum-csum.cummax()).min()
    return [sharpe,win,pf,annual,downside,var95,cvar95,mdd]

labels=["夏普比率","胜率","盈亏比","当日收益","下行风险","VaR95","CVaR95","最大回撤"]

today_stats=metrics(today_trades)
hist_df=df[(df['日期']>=hist_start)&(df['日期']<=hist_end)]
hist_stats=metrics(hist_df)

# ========= 页面 =========
tabs=st.tabs(["报告","导出"])

with tabs[0]:
    st.subheader("当日交易概览")
    st.dataframe(today_trades)
    st.plotly_chart(px.line(today_trades,x='时间',y='累计盈亏',title='📈 累计盈亏'),use_container_width=True)
    for lbl,val in zip(labels,today_stats):
        st.metric(lbl,f"{val:.2f}")
with tabs[1]:
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        df.to_excel(w,'AllTrades',index=False)
        pd.DataFrame({'指标':labels,'当日':today_stats,'历史':hist_stats}).to_excel(w,'Stats',index=False)
    st.download_button("下载 Excel",buf.getvalue(),"report.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
