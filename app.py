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
    layout="wide", initial_sidebar_state="expanded", page_icon="📊"
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

# 快照管理
def manage_snapshots(df):
    SNAP_DIR = 'snapshots'; os.makedirs(SNAP_DIR, exist_ok=True)
    snap = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.csv"; df.to_csv(os.path.join(SNAP_DIR, snap), index=False)
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
df['累计盈亏']=df['盈亏'].cumsum(); df['日期']=df['时间'].dt.date; df['小时']=df['时间'].dt.hour

# 统计函数
def compute_stats(data, lookback):
    total_trades=len(data); total_pl=data['盈亏'].sum(); avg_pl=data['盈亏'].mean() if total_trades else 0
    cust=cumsum=cumsum=pd.Series(data['盈亏']).cumsum()
    max_dd=(cumsum - cumsum.cummax()).min() if total_trades else 0
    rel=(cumsum - cumsum.cummax())/cumsum.cummax(); max_rel=rel.min() if total_trades else 0
    pf=data[data['盈亏']>0]['盈亏'].sum() / -data[data['盈亏']<0]['盈亏'].sum() if total_trades and data['盈亏'].min()<0 else np.nan
    profit_rate= total_pl/data['数量'].sum() if data['数量'].sum() else np.nan
    win_cnt=(data['盈亏']>0).sum(); win_sum=data[data['盈亏']>0]['盈亏'].sum(); win_avg=data[data['盈亏']>0]['盈亏'].mean() if win_cnt else 0
    loss_cnt=(data['盈亏']<0).sum(); loss_sum=data[data['盈亏']<0]['盈亏'].sum(); loss_avg=data[data['盈亏']<0]['盈亏'].mean() if loss_cnt else 0
    total_days=data['日期'].nunique(); win_days=data[data['盈亏']>0]['日期'].nunique(); loss_days=data[data['盈亏']<0]['日期'].nunique()
    total_comm=data['手续费'].sum()
    # 核心额外指标
    sharpe=data['盈亏'].mean()/data['盈亏'].std()*np.sqrt(252) if data['盈亏'].std() else np.nan
    winrate=(data['盈亏']>0).mean(); pf_ratio=pf; ann_ret=total_pl/((data['时间'].max()-data['时间'].min()).days or 1)*252
    downside=data[data['盈亏']<0]['盈亏'].std(); var95=-data['盈亏'].quantile(0.05)
    cvar95=-data[data['盈亏']<=data['盈亏'].quantile(0.05)]['盈亏'].mean()
    roll_max= cumsum.rolling(window=lookback).max(); dd_lookback=(cumsum-roll_max).min()
    return [total_trades,total_pl,avg_pl,max_dd,max_rel,pf,profit_rate,win_cnt,win_sum,win_avg,loss_cnt,loss_sum,loss_avg,total_days,win_days,loss_days,total_comm,sharpe,winrate,pf_ratio,ann_ret,downside,var95,cvar95,dd_lookback]

# UI布局
tabs=st.tabs(['报告视图','数据导出','⚙️ 设置'])
with tabs[0]:
    st.subheader('📅 当日成交明细'); st.dataframe(trades_today)
    st.subheader('📌 当日 & 历史统计指标')
    labels=['交易总笔数','总盈亏','平均盈亏','最大回撤','最大相对跌幅','利润系数','利润率','盈利交易','盈利总计','平均利润','亏损交易','亏损总额','平均亏损','总天数','盈利天数','亏损天数','手续费','夏普率','胜率','盈亏比','年化收益率','下行风险','VaR95','CVaR95',f'{lookback_days}天回撤']
    today_vals=compute_stats(trades_today,lookback_days); hist_vals=compute_stats(df,lookback_days)
    for title,vals in [('当日统计指标',today_vals),('历史统计指标',hist_vals)]:
        st.markdown(f"### {title}")
        cols=st.columns(4)
        for i,(lbl,val) in enumerate(zip(labels,vals)):
            cols[i%4].metric(lbl,f'{val:.2f}' if isinstance(val,(float,np.floating)) else str(val))
with tabs[1]:
    # 数据导出省略...
    pass
with tabs[2]:
    st.subheader('⚙️ 设置')
    st.markdown(
        '- **缓存天数（天）**: 侧边栏输入框  
        - **保留快照份数**: 侧边栏输入框  
        - **回撤回溯期 (天)**: 侧边栏滑动条'
    )
