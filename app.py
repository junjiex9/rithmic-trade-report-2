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

# ============ 侧边栏上传与设置 ============
st.sidebar.header('📁 上传与设置')
uploaded = st.sidebar.file_uploader('上传交易 CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('市场快照 CSV', type='csv')
sent_file = st.sidebar.file_uploader('舆情数据 CSV', type='csv')
max_snapshots = st.sidebar.number_input('保留快照份数', min_value=1, value=10)
cache_days = st.sidebar.number_input('缓存天数（天）', min_value=1, value=1)
lookback_days = st.sidebar.slider('回撤回溯期 (天)', 1, 60, 30)

SNAP_DIR = 'snapshots'
os.makedirs(SNAP_DIR, exist_ok=True)

# ============ 数据加载 ============
@st.cache_data(show_spinner=False, max_entries=10, ttl=3600*24*cache_days)
def load_and_clean(files):
    def extract_orders(f):
        content = f.getvalue().decode('utf-8', errors='ignore').splitlines()
        idx = next((i for i,l in enumerate(content) if 'Completed Orders' in l), None)
        if idx is None: return pd.DataFrame()
        header = content[idx+1].replace('"','').split(',')
        body   = '\n'.join(content[idx+2:])
        df = pd.read_csv(io.StringIO(body), names=header)
        df['上传文件'] = f.name
        return df
    dfs = [extract_orders(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Status']=='Filled']
    df = df[['Account','Buy/Sell','Symbol','Avg Fill Price','Qty To Fill',
             'Update Time (CST)','Commission Fill Rate','Closed Profit/Loss','上传文件']]
    df.columns = ['账户','方向','品种','价格','数量','时间','手续费','盈亏','上传文件']
    df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
    df['方向'] = df['方向'].map({'B':'Buy','S':'Sell'})
    for c in ['价格','数量','手续费','盈亏']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['时间','方向']).sort_values('时间').reset_index(drop=True)

if not uploaded:
    st.info('👆 请上传交易CSV以开始分析。')
    st.stop()

df = load_and_clean(uploaded)

# 基本衍生字段
df['累计盈亏'] = df['盈亏'].cumsum()
df['日期'] = df['时间'].dt.date
df['小时'] = df['时间'].dt.hour

# 核心指标
period_days = max((df['时间'].max() - df['时间'].min()).days, 1)
total_pl = df['盈亏'].sum()
ann_return = total_pl/period_days*252
sharpe = df['盈亏'].mean()/df['盈亏'].std()*np.sqrt(252)
winrate = (df['盈亏']>0).mean()
profit_factor = (df[df['盈亏']>0]['盈亏'].mean()
                 / -df[df['盈亏']<0]['盈亏'].mean())
mdd = (df['累计盈亏'] - df['累计盈亏'].cummax()).min()
calmar = ann_return/abs(mdd) if mdd<0 else np.nan
roll_max = df['累计盈亏'].rolling(window=lookback_days).max()
recent_dd = (df['累计盈亏'] - roll_max).min()

# 生成图表
sections = []
# 1. 📈 累计盈亏趋势
sections.append(('📈 累计盈亏趋势', [px.line(df, x='时间', y='累计盈亏')]))
# 2. 📊 日/小时盈亏
fig_day = px.bar(df.groupby('日期')['盈亏'].sum().reset_index(),
                 x='日期', y='盈亏')
fig_hour = px.bar(df.groupby('小时')['盈亏'].mean().reset_index(),
                   x='小时', y='盈亏')
sections.append(('📊 日/小时盈亏', [fig_day, fig_hour]))
# 3. ⏳ 持仓时长分布（分钟）
df_sorted = df.sort_values(['账户','品种','时间'])
df_sorted['持仓时长'] = (df_sorted.groupby(['账户','品种'])['时间']
                         .diff().dt.total_seconds()/60)
fig_acc = px.box(df_sorted, x='账户', y='持仓时长')
fig_sym = px.box(df_sorted, x='品种', y='持仓时长')
sections.append(('⏳ 持仓时长分布（分钟）', [fig_acc, fig_sym]))
# 4. 🎲 Monte Carlo 模拟
returns = df['盈亏'].values
final = [np.random.choice(returns, len(returns), replace=True).cumsum()[-1]
         for _ in range(500)]
sections.append(('🎲 Monte Carlo 模拟', [px.histogram(final, nbins=50)]))
# 5. 🕳️ 滑点与成交率分析
if market_file:
    mp = pd.read_csv(market_file)
    mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
    mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'}, inplace=True)
    dfm = pd.merge(df, mp, left_on=['品种','时间'],
                   right_on=['品种','Time'], how='left')
    dfm['滑点'] = dfm['价格'] - dfm['市场价格']
    sections.append(('🕳️ 滑点与成交率分析',
                     [px.histogram(dfm, x='滑点')]))
# 6. 📣 社交舆情热力图
if sent_file:
    ds = pd.read_csv(sent_file)
    ds['Date'] = pd.to_datetime(ds['Date'], errors='coerce').dt.date
    heat = ds.pivot_table(values='SentimentScore',
                          index='Symbol', columns='Date',
                          aggfunc='mean')
    sections.append(('📣 社交舆情热力图',
                     [px.imshow(heat, aspect='auto')]))

# UI布局
tabs = st.tabs(['报告视图','数据导出','设置'])
with tabs[0]:
    for title, figs in sections:
        st.subheader(title)
        for fig in figs:
            st.plotly_chart(fig, use_container_width=True)
    st.subheader('📌 核心统计指标')
    cols = st.columns(6)
    cols[0].metric('夏普比率', f"{sharpe:.2f}")
    cols[1].metric('胜率', f"{winrate:.2%}")
    cols[2].metric('盈亏比', f"{profit_factor:.2f}")
    cols[3].metric('最大回撤', f"{mdd:.2f}")
    cols[4].metric(f"{lookback_days}天回撤", f"{recent_dd:.2f}")
    cols[5].metric('Calmar 比率', f"{calmar:.2f}")

with tabs[1]:
    # 导出 Excel
    if st.button('导出 Excel (.xlsx)'):
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            df.to_excel(w, sheet_name='Trades', index=False)
            metrics = pd.DataFrame({
                '指标':['夏普比率','胜率','盈亏比','最大回撤',
                        f'{lookback_days}天回撤','Calmar 比率'],
                '数值':[sharpe,winrate,profit_factor,mdd,
                        recent_dd,calmar]
            })
            metrics.to_excel(w, sheet_name='Metrics', index=False)
        st.download_button('下载 Excel', buf.getvalue(),
                           file_name='report.xlsx')
    # 导出 PDF
    if st.button('导出 PDF 报告'):
        pdf = FPDF('P','mm','A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()
        # 封面
        pdf.add_page()
        pdf.set_font('Arial','B',24)
        pdf.cell(0,60,'',ln=1)
        pdf.cell(0,10,'📈 交易分析报告',ln=1,align='C')
        pdf.set_font('Arial','',12)
        pdf.cell(0,10,f'生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}',ln=1,align='C')
        # 目录
        pdf.add_page()
        pdf.set_font('Arial','B',16)
        pdf.cell(0,10,'目录',ln=1)
        pdf.set_font('Arial','',12)
        for i, (title, _) in enumerate(sections+ [('📌 核心统计指标', [])], 1):
            pdf.cell(0,8,f'{i}. {title}',ln=1)
        # 内容
        for title, figs in sections:
            pdf.add_page()
            pdf.set_font('Arial','B',14)
            pdf.cell(0,10,title,ln=1)
            for fig in figs:
                img = fig.to_image(format='png', width=600, height=300)
                pdf.image(io.BytesIO(img), x=15, y=pdf.get_y()+5, w=180)
        # 核心指标页
        pdf.add_page()
        pdf.set_font('Arial','B',14)
        pdf.cell(0,10,'📌 核心统计指标',ln=1)
        pdf.set_font('Arial','',12)
        items = [
            ('夏普比率', f"{sharpe:.2f}"),
            ('胜率', f"{winrate:.2%}"),
            ('盈亏比', f"{profit_factor:.2f}"),
            ('最大回撤', f"{mdd:.2f}"),
            (f"{lookback_days}天回撤", f"{recent_dd:.2f}"),
            ('Calmar 比率', f"{calmar:.2f}")
        ]
        for name, val in items:
            pdf.cell(60,8,name)
            pdf.cell(0,8,val,ln=1)
        # 页脚
        total_pages = pdf.page_no()
        for p in range(1, total_pages+1):
            pdf.page = p
            pdf.set_y(-15)
            pdf.set_font('Arial','I',8)
            pdf.cell(0,10,f'第 {p}/{total_pages} 页',align='C')
        data = pdf.output(dest='S').encode('latin1')
        st.download_button('下载 PDF', data, 'report.pdf')
with tabs[2]:
    st.subheader('⚙️ 设置')
    st.write('缓存天数、快照保留、回撤回溯期等参数。')
