import streamlit as st
import pandas as pd
import numpy as np
import io, os
import plotly.express as px
from datetime import datetime

# ============ 页面配置 ============
st.set_page_config(page_title="📈 自动化交易分析报告生成器", layout="wide", initial_sidebar_state="expanded")

# ============ 多语言支持 ============
LANG = {'中文': '📈 自动化交易分析报告生成器', 'English': '📈 Automated Trading Report Generator'}
lang = st.sidebar.selectbox('语言 / Language', list(LANG.keys()))
st.title(LANG[lang])

# ============ 侧边栏 ============
st.sidebar.header('📁 上传与快照管理')
uploaded = st.sidebar.file_uploader('上传 Rithmic/ATAS 导出 CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('上传市场快照 CSV (Symbol,Time,MarketPrice)', type='csv')
sent_file = st.sidebar.file_uploader('上传舆情数据 CSV (Symbol,Date,SentimentScore)', type='csv', key='sentiment')
max_snapshots = st.sidebar.number_input('保留最近快照份数', min_value=1, value=10)

SNAP_DIR = 'snapshots'
os.makedirs(SNAP_DIR, exist_ok=True)

# ============ 数据加载 ============
@st.cache_data
def load_and_clean(files):
    def extract_orders(f):
        content = f.getvalue().decode('utf-8', errors='ignore').splitlines()
        idx = next((i for i,l in enumerate(content) if 'Completed Orders' in l), None)
        if idx is None: return pd.DataFrame()
        header = content[idx+1].replace('\"','').split(',')
        body   = '\n'.join(content[idx+2:])
        df = pd.read_csv(io.StringIO(body), names=header)
        df['上传文件'] = f.name
        return df
    dfs = [extract_orders(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Status'] == 'Filled']
    cols = ['Account','Buy/Sell','Symbol','Avg Fill Price','Qty To Fill',
            'Update Time (CST)','Commission Fill Rate','Closed Profit/Loss','上传文件']
    df = df[cols].rename(columns={
        'Account':'账户','Buy/Sell':'方向','Symbol':'品种','Avg Fill Price':'价格',
        'Qty To Fill':'数量','Update Time (CST)':'时间','Commission Fill Rate':'手续费',
        'Closed Profit/Loss':'盈亏','上传文件':'上传文件'
    })
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['方向'] = df['方向'].map({'B':'Buy','S':'Sell'})
    for c in ['价格','数量','手续费','盈亏']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['时间','价格','方向']).sort_values('时间').reset_index(drop=True)
    return df

if uploaded:
    df = load_and_clean(uploaded)

    # 保存并清理快照
    snap_file = f"snapshot_{len(uploaded)}files_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(os.path.join(SNAP_DIR, snap_file), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    if len(snaps) > max_snapshots:
        for old in snaps[:-max_snapshots]:
            os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，快照：{snap_file}")
    st.sidebar.write({f.name: len(df[df['上传文件']==f.name]) for f in uploaded})

    view = st.sidebar.selectbox('视图分组', ['总体','按账户','按品种'])
    st.sidebar.header('⚠️ 风险阈值预警')
    max_loss = st.sidebar.number_input('单笔最大亏损', value=-100.0)
    max_trades = st.sidebar.number_input('日内最大交易次数', value=50)
    if df['盈亏'].min() < max_loss:
        st.warning(f"⚠️ 存在单笔盈亏低于阈值({max_loss})！")
    if df[df['时间'].dt.date == datetime.today().date()].shape[0] > max_trades:
        st.warning(f"⚠️ 今日交易次数超过阈值({max_trades})！")

    # 基本指标
    df['累计盈亏'] = df['盈亏'].cumsum()
    df['日期'] = df['时间'].dt.date
    df['小时'] = df['时间'].dt.hour
    period_days = (df['时间'].max() - df['时间'].min()).days or 1
    total_pl = df['盈亏'].sum()
    ann_return = total_pl / period_days * 252
    downside_dev = df[df['盈亏'] < 0]['盈亏'].std()
    var95 = -df['盈亏'].quantile(0.05)
    cvar95 = -df[df['盈亏'] <= df['盈亏'].quantile(0.05)]['盈亏'].mean()
    sharpe = df['盈亏'].mean() / df['盈亏'].std() * np.sqrt(252)
    winrate = (df['盈亏'] > 0).mean()
    profit_factor = df[df['盈亏'] > 0]['盈亏'].mean() / (-df[df['盈亏'] < 0]['盈亏'].mean())
    mdd = (df['累计盈亏'] - df['累计盈亏'].cummax()).min()

    # 交互式图表
    st.subheader('📈 累计盈亏趋势')
    if view == '按账户':
        fig = px.line(df, x='时间', y='累计盈亏', color='账户')
    elif view == '按品种':
        fig = px.line(df, x='时间', y='累计盈亏', color='品种')
    else:
        fig = px.line(df, x='时间', y='累计盈亏')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('📊 日/小时盈亏')
    st.plotly_chart(px.bar(df.groupby('日期')['盈亏'].sum().reset_index(), x='日期', y='盈亏'))
    st.plotly_chart(px.bar(df.groupby('小时')['盈亏'].mean().reset_index(), x='小时', y='盈亏'))

    # 持仓时长分布
    st.subheader('⏳ 持仓时长分布（分钟）')
    df_sorted = df.sort_values(['账户','品种','时间'])
    df_sorted['持仓时长'] = df_sorted.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60
    st.plotly_chart(px.box(df_sorted, x='账户', y='持仓时长', title='按账户持仓时长'))
    st.plotly_chart(px.box(df_sorted, x='品种', y='持仓时长', title='按品种持仓时长'))

    # Monte Carlo
    st.subheader('🎲 Monte Carlo 模拟')
    returns = df['盈亏'].values
    sims, n = 500, len(returns)
    final = [np.random.choice(returns, n, replace=True).cumsum()[-1] for _ in range(sims)]
    st.plotly_chart(px.histogram(final, nbins=40, title='Monte Carlo 累积盈亏'))

    # 滑点分析
    st.subheader('🕳️ 滑点与成交率分析')
    if market_file:
        mp = pd.read_csv(market_file)
        mp['Time']=pd.to_datetime(mp['Time'], errors='coerce')
        mp = mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'})
        df = df.merge(mp, left_on=['品种','时间'], right_on=['品种','Time'], how='left')
        df['滑点']=df['价格']-df['市场价格']
        st.plotly_chart(px.histogram(df, x='滑点', nbins=50, title='滑点分布'))
    else:
        st.info('请在侧边栏上传市场快照CSV以启用滑点分析')

    # 舆情热力图
    st.subheader('📣 社交舆情热力图')
    if sent_file:
        df_sent = pd.read_csv(sent_file)
        df_sent['Date']=pd.to_datetime(df_sent['Date'], errors='coerce').dt.date
        heat = df_sent.pivot_table(index='Symbol', columns='Date', values='SentimentScore', aggfunc='mean')
        st.plotly_chart(px.imshow(heat, aspect='auto', title='舆情热力图'))
    else:
        st.info('请在侧边栏上传舆情CSV以启用热力图')

    # 核心指标
    st.subheader('📌 核心统计指标')
    st.metric('夏普比率', f"{sharpe:.2f}")
    st.metric('胜率', f"{winrate:.2%}")
    st.metric('盈亏比', f"{profit_factor:.2f}")
    st.metric('年化收益率', f"{ann_return:.2f}")
    st.metric('下行风险', f"{downside_dev:.2f}")
    st.metric('VaR95', f"{var95:.2f}")
    st.metric('CVaR95', f"{cvar95:.2f}")
    st.metric('最大回撤', f"{mdd:.2f}")

    if st.button('📄 导出PDF报告'):
        st.info('PDF导出功能待实现')
else:
    st.info('👆 请在侧边栏上传CSV文件进行分析')
