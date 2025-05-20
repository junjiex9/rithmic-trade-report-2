import streamlit as st
import pandas as pd
import numpy as np
import io, os
import plotly.express as px
import plotly.io as pio
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
for key, default in [('cache_days', DEFAULT_CACHE_DAYS), ('max_snapshots', DEFAULT_MAX_SNAPSHOTS), ('lookback_days', DEFAULT_LOOKBACK)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============ 侧边栏上传与设置 ============
st.sidebar.header('📁 上传与设置')
uploaded = st.sidebar.file_uploader('上传交易 CSV', type='csv', accept_multiple_files=True)
market_file = st.sidebar.file_uploader('市场快照 CSV', type='csv')
sent_file = st.sidebar.file_uploader('舆情数据 CSV', type='csv')

# 读取动态设置
cache_days = st.sidebar.number_input('缓存天数（天）', min_value=1, value=st.session_state['cache_days'])
max_snapshots = st.sidebar.number_input('保留快照份数', min_value=1, value=st.session_state['max_snapshots'])
lookback_days = st.sidebar.slider('回撤回溯期 (天)', 1, 60, value=st.session_state['lookback_days'])
# 写回会话状态
st.session_state['cache_days'] = cache_days
st.session_state['max_snapshots'] = max_snapshots
st.session_state['lookback_days'] = lookback_days

if not uploaded:
    st.sidebar.info('请上传交易CSV以开始。')
    st.stop()

# ============ 风险阈值预警 ============
st.sidebar.header('⚠️ 风险阈值预警')
max_loss = st.sidebar.number_input('单笔最大亏损', value=-100.0)
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

# 加载并缓存数据
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

# 风险警示
if '盈亏' in df.columns:
    if df['盈亏'].min() < max_loss:
        st.warning(f"⚠️ 存在单笔盈亏低于阈值({max_loss})！最小盈亏：{df['盈亏'].min():.2f}")
    today_trades = df[df['时间'].dt.date == datetime.today().date()].shape[0]
    if today_trades > max_trades:
        st.warning(f"⚠️ 今日交易次数 {today_trades} 超过阈值({max_trades})！")

# 衍生字段
df['累计盈亏'] = df['盈亏'].cumsum()
df['日期'] = df['时间'].dt.date
df['小时'] = df['时间'].dt.hour

# 指标计算
def compute_metrics(lookback):
    period_days = max((df['时间'].max() - df['时间'].min()).days, 1)
    total_pl = df['盈亏'].sum()
    sharpe = df['盈亏'].mean() / df['盈亏'].std() * np.sqrt(252)
    winrate = (df['盈亏'] > 0).mean()
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
    # 累计盈亏趋势
    st.subheader('📈 累计盈亏趋势')
    fig1 = px.line(df, x='时间', y='累计盈亏', title='累计盈亏趋势')
    st.plotly_chart(fig1, use_container_width=True)

    # 日/小时盈亏
    st.subheader('📊 日/小时盈亏')
    fig2 = px.bar(df.groupby('日期')['盈亏'].sum().reset_index(), x='日期', y='盈亏', title='每日盈亏')
    fig3 = px.bar(df.groupby('小时')['盈亏'].mean().reset_index(), x='小时', y='盈亏', title='每小时平均盈亏')
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    # 持仓时长分布
    st.subheader('⏳ 持仓时长分布（分钟）')
    sorted_df = df.sort_values(['账户','品种','时间'])
    sorted_df['持仓时长'] = sorted_df.groupby(['账户','品种'])['时间'].diff().dt.total_seconds()/60
    fig4 = px.box(sorted_df, x='账户', y='持仓时长', title='按账户')
    fig5 = px.box(sorted_df, x='品种', y='持仓时长', title='按品种')
    st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig5, use_container_width=True)

    # Monte Carlo 模拟
    st.subheader('🎲 Monte Carlo 模拟')
    sims = [np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1] for _ in range(500)]
    fig6 = px.histogram(sims, nbins=40, title='Monte Carlo 累积盈亏分布')
    st.plotly_chart(fig6, use_container_width=True)

    # 滑点与成交率
    if market_file:
        st.subheader('🕳️ 滑点与成交率分析')
        mp = pd.read_csv(market_file)
        mp['Time'] = pd.to_datetime(mp['Time'], errors='coerce')
        mp.rename(columns={'MarketPrice':'市场价格','Symbol':'品种'}, inplace=True)
        merged = df.merge(mp, left_on=['品种','时间'], right_on=['品种','Time'], how='left')
        merged['滑点'] = merged['价格'] - merged['市场价格']
        fig7 = px.histogram(merged, x='滑点', nbins=50, title='滑点分布')
        st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info('请上传市场快照 CSV 以查看滑点分析')

    # 社交舆情热力图
    if sent_file:
        st.subheader('📣 社交舆情热力图')
        ds = pd.read_csv(sent_file)
        ds['Date'] = pd.to_datetime(ds['Date'], errors='coerce').dt.date
        heat = ds.pivot_table(values='SentimentScore', index='Symbol', columns='Date', aggfunc='mean')
        fig8 = px.imshow(heat, aspect='auto', title='舆情热力图')
        st.plotly_chart(fig8, use_container_width=True)
    else:
        st.info('请上传舆情数据 CSV 以查看热力图')

    # 核心统计指标
    st.subheader('📌 核心统计指标')
    sharpe, winrate, pf, mdd, calmar, recent_dd = compute_metrics(lookback_days)
    cols = st.columns(6)
    cols[0].metric('夏普率', f"{sharpe:.2f}")
    cols[1].metric('胜率', f"{winrate:.2%}")
    cols[2].metric('盈亏比', f"{pf:.2f}")
    cols[3].metric('最大回撤', f"{mdd:.2f}")
    cols[4].metric(f"{lookback_days}天回撤", f"{recent_dd:.2f}")
    cols[5].metric('Calmar 比率', f"{calmar:.2f}")

# 2. 数据导出
with tabs[1]:
    st.subheader('📤 数据导出')
    col_excel, col_pdf = st.columns(2)

    # 下载 Excel (.xlsx)
    with col_excel:
        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Trades', index=False)
            pd.DataFrame({
                '指标': ['总交易次数','总盈亏','夏普率','胜率','盈亏比','最大回撤','Calmar','回撤(天)'],
                '数值': [len(df), df['盈亏'].sum(), *compute_metrics(lookback_days)]
            }).to_excel(writer, sheet_name='Metrics', index=False)
        st.download_button(
            label='下载 Excel (.xlsx)',
            data=excel_buf.getvalue(),
            file_name='report.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    # 下载 PDF 报告
    with col_pdf:
        sims_pdf = [np.random.choice(df['盈亏'], len(df), replace=True).cumsum()[-1] for _ in range(500)]
        pdf = FPDF('P','mm','A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()
        pdf.set_font('Helvetica', '', 12)
        pdf.add_page()
        pdf.cell(0,60, '', ln=1)
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0,10, 'Automated Trading Analysis Report', ln=1, align='C')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0,10, f'Generated: {datetime.now():%Y-%m-%d %H:%M:%S}', ln=1, align='C')
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0,10, 'Core Metrics', ln=1)
        pdf.set_font('Helvetica', '', 12)
        labels = ['Total Trades','Total P/L','Sharpe Ratio','Win Rate','Profit Factor','Max Drawdown','Calmar Ratio','Recent Drawdown']
        values = [len(df), df['盈亏'].sum()] + list(compute_metrics(lookback_days))
        for label, val in zip(labels, values):
            pdf.cell(60,8, label)
            pdf.cell(0,8, f"{val:.2f}" if isinstance(val, float) else str(val), ln=1)
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0,10, 'Monte Carlo Distribution', ln=1)
        pdf.set_font('Helvetica', '', 12)
        mc_fig = px.histogram(sims_pdf, nbins=40)
mc_img = mc_fig.to_image(format='png', width=600, height=300, engine='kaleido')
# 写入临时PNG并载入到PDF
img_path = 'temp_mc.png'
with open(img_path, 'wb') as imgf:
    imgf.write(mc_img)
pdf.image(img_path, x=15, y=pdf.get_y()+5, w=180)
os.remove(img_path)(io.BytesIO(mc_img), x=15, y=pdf.get_y()+5, w=180)
        tmp_path = 'temp_report.pdf'
        pdf.output(tmp_path)
        with open(tmp_path, 'rb') as f:
            pdf_bytes = f.read()
        st.download_button(
            label='下载 PDF 报告',
            data=pdf_bytes,
            file_name='report.pdf',
            mime='application/pdf'
        )

# 3. 设置
with tabs[2]:
    st.subheader('⚙️ 设置')
    st.markdown(
        '''
        通过右侧侧边栏直接调整以下参数：
        - **缓存天数（天）**: 控制数据缓存过期时间
        - **保留快照份数**: 控制历史快照最大保留数量
        - **回撤回溯期 (天)**: 控制最大回撤计算的回溯窗口
        '''
    )
    st.info('修改后请在侧边栏重新运行或刷新页面以生效。')
