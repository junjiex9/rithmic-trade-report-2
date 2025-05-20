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
DEFAULT_CACHE_DAYS = 1
DEFAULT_MAX_SNAPSHOTS = 10
DEFAULT_LOOKBACK = 30

# 初始化会话状态
for key, default in [
    ('cache_days', DEFAULT_CACHE_DAYS),
    ('max_snapshots', DEFAULT_MAX_SNAPSHOTS),
    ('lookback_days', DEFAULT_LOOKBACK)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============ 侧边栏上传与设置 ============
st.sidebar.header('📁 上传与设置')
uploaded = st.sidebar.file_uploader('上传交易 CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('市场快照 CSV', type='csv')
sent_file = st.sidebar.file_uploader('舆情数据 CSV', type='csv')

# 读取并更新动态设置
cache_days = st.sidebar.number_input('缓存天数（天）', min_value=1, value=st.session_state['cache_days'])
max_snapshots = st.sidebar.number_input('保留快照份数', min_value=1, value=st.session_state['max_snapshots'])
lookback_days = st.sidebar.slider('回撤回溯期 (天)', 1, 60, value=st.session_state['lookback_days'])
st.session_state['cache_days'] = cache_days
st.session_state['max_snapshots'] = max_snapshots
st.session_state['lookback_days'] = lookback_days

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
    def extract_orders(f):
        content = f.getvalue().decode('utf-8', errors='ignore').splitlines()
        idx = next((i for i, l in enumerate(content) if 'Completed Orders' in l), None)
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

# 加载数据
import pandas as pd

df = load_and_clean(uploaded)

# 快照管理
def manage_snapshots(df):
    SNAP_DIR = 'snapshots'
    os.makedirs(SNAP_DIR, exist_ok=True)
    snap_file = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(os.path.join(SNAP_DIR, snap_file), index=False)
    snaps = sorted(os.listdir(SNAP_DIR))
    if len(snaps) > max_snapshots:
        for old in snaps[:-max_snapshots]:
            os.remove(os.path.join(SNAP_DIR, old))
    st.sidebar.success(f"已加载 {len(df)} 条交易，快照：{snap_file}")

manage_snapshots(df)

# ============ 风险警示 ============
if isinstance(df, pd.DataFrame) and '盈亏' in df.columns:
    # 单笔亏损预警
    over_loss = df[df['盈亏'] <= -abs(max_loss)]
    if not over_loss.empty:
        st.sidebar.warning(f"发现 {len(over_loss)} 笔亏损超过阈值 ({max_loss})！")
    # 日内交易次数预警
    today = datetime.now().date()
    trades_today = df[df['时间'].dt.date == today]
    if len(trades_today) > max_trades:
        st.sidebar.warning(f"今日交易次数 {len(trades_today)} 次，超过阈值 ({max_trades})！")

# 衍生字段
df['累计盈亏'] = df['盈亏'].cumsum()
df['日期'] = df['时间'].dt.date
df['小时'] = df['时间'].dt.hour

# 指标计算
def compute_metrics(lookback):
    period_days = max((df['时间'].max() - df['时间'].min()).days, 1)
    total_pl = df['盈亏'].sum()
    sharpe = df['盈亏'].mean() / df['盈亏'].std() * np.sqrt(252)
    winrate = (df['盈亏']>0).mean()
    pf = df[df['盈亏']>0]['盈亏'].mean() / -df[df['盈亏']<0]['盈亏'].mean()
    mdd = (df['累计盈亏'] - df['累计盈亏'].cummax()).min()
    calmar = total_pl/period_days*252/abs(mdd) if mdd<0 else np.nan
    roll_max = df['累计盈亏'].rolling(window=lookback).max()
    recent_dd = (df['累计盈亏'] - roll_max).min()
    return sharpe, winrate, pf, mdd, calmar, recent_dd

# UI布局
tabs = st.tabs(['报告视图','数据导出','⚙️ 设置'])

# 1. 报告视图
with tabs[0]:
    st.subheader('📈 累计盈亏趋势')
    st.plotly_chart(px.line(df, x='时间', y='累计盈亏'), use_container_width=True)
    # … 其余图表省略 …
    st.subheader('📌 核心指标')
    sharpe, winrate, pf, mdd, calmar, recent_dd = compute_metrics(lookback_days)
    cols = st.columns(6)
    cols[0].metric('夏普率', f"{sharpe:.2f}")
    cols[1].metric('胜率', f"{winrate:.2%}")
    cols[2].metric('盈亏比', f"{pf:.2f}")
    cols[3].metric('最大回撤', f"{mdd:.2f}")
    cols[4].metric(f"{lookback_days}天回撤", f"{recent_dd:.2f}")
    cols[5].metric('Calmar', f"{calmar:.2f}")

# 2. 数据导出
with tabs[1]:
    st.subheader('📤 数据导出')
    col_excel, col_pdf = st.columns(2)
    with col_excel:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Trades', index=False)
            pd.DataFrame({
                '指标':['总交易次数','总盈亏','夏普率','胜率','盈亏比','最大回撤','Calmar','回撤(天)'],
                '数值':[len(df), df['盈亏'].sum(), *compute_metrics(lookback_days)]
            }).to_excel(writer, sheet_name='Metrics', index=False)
        st.download_button('下载 Excel', excel_buf.getvalue(), file_name='report.xlsx')
    with col_pdf:
        sims = [np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1] for _ in range(500)]
        pdf = FPDF('P','mm','A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()
        pdf.set_font('Helvetica','',12)
        pdf.add_page()
        pdf.cell(0,60,'',ln=1)
        pdf.set_font('Helvetica','B',16)
        pdf.cell(0,10,'Automated Trading Analysis Report',ln=1,align='C')
        pdf.set_font('Helvetica','',10)
        pdf.cell(0,10,f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}',ln=1,align='C')
        pdf.add_page()
        pdf.set_font('Helvetica','B',14)
        pdf.cell(0,10,'Core Metrics',ln=1)
        pdf.set_font('Helvetica','',12)
        labels = ['Total Trades','Total P/L','Sharpe Ratio','Win Rate','Profit Factor','Max Drawdown','Calmar Ratio','Recent Drawdown']
        vals = [len(df), df['盈亏'].sum(), *compute_metrics(lookback_days)]
        for lbl, v in zip(labels, vals):
            pdf.cell(60,8,lbl); pdf.cell(0,8,f"{v:.2f}" if isinstance(v,float) else str(v),ln=1)
        pdf.add_page()
        pdf.set_font('Helvetica','B',14)
        pdf.cell(0,10,'Monte Carlo Distribution',ln=1)
        pdf.set_font('Helvetica','',12)
        img = px.histogram(sims, nbins=40).to_image(format='png', width=600, height=300)
        pdf.image(io.BytesIO(img), x=15, y=pdf.get_y()+5, w=180)
        tmp = 'tmp.pdf'
        pdf.output(tmp)
        pdf_bytes = open(tmp,'rb').read()
        st.download_button('下载 PDF 报告', pdf_bytes, file_name='report.pdf', mime='application/pdf')

# 3. 设置
with tabs[2]:
    st.subheader('⚙️ 设置')
    st.markdown(
        '''
- **缓存天数（天）**: 侧边栏的输入框用于设置数据缓存的有效期
- **保留快照份数**: 侧边栏的输入框用于控制保存的快照数量
- **回撤回溯期 (天)**: 侧边栏的滑动条用于调整最大回撤计算的窗口期
        '''
    )
    st.info('修改后请刷新或重新运行应用以生效。')
