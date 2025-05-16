# 交易分析报告生成器 📈

本项目基于 Streamlit 搭建，提供完整的 Rithmic/ATAS 交易数据**复盘分析**与**报告导出**功能，包含：

* **数据清洗**：自动识别并加载「Completed Orders」表格，合并多文件成交记录
* **核心统计**：Total PnL、Sharpe Ratio、Win Rate、Profit Factor、Annual Return、Downside Deviation、VaR95、CVaR95、Max Drawdown
* **图表展示**：

  * 累计盈亏趋势折线图
  * 日/小时盈亏柱状图
  * 持仓时长分布箱线图
  * Monte Carlo 模拟分布图
  * 滑点分布直方图（可选）
  * 社交舆情热力图（可选）
* **导出报告**：

  * **Excel**：Trades、DailyPL、HourlyPL、AccountStats、SymbolStats、MonthlyPL、Durations、Summary 等详细工作表
  * **PDF**：封面 + 核心统计 + 多维度表格（账户/品种/月度/持仓/模拟/舆情）
* **多语言支持**：中/英界面切换
* **快照管理**：自动保存/清理历史数据快照

---

## 目录结构

```
├── app.py                 # 主应用脚本
├── app_pdf.py             # （可选）PDF导出封装模块
├── requirements.txt       # 依赖列表
├── README.md              # 项目说明
├── snapshots/             # 数据快照目录（自动生成）
└── sent_heat.png          # 舆情热力图（运行时生成）
```

## 安装与运行

1. 克隆仓库：

   ```bash
   git clone <仓库地址>
   cd <项目目录>
   ```
2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```
3. 本地运行：

   ```bash
   streamlit run app.py
   ```

## 在 Streamlit Cloud 部署

1. 将项目推送到 GitHub 仓库
2. 登录 [Streamlit Cloud](https://streamlit.io/cloud)，新建 App，选择对应仓库和 `app.py`
3. 部署完成后，即可通过在线链接访问

## 自定义与扩展

* 若需自定义 PDF 格式，可在 `app_pdf.py` 中调整 `write_full_pdf` 函数
* 为支持舆情图像导出，请安装 `kaleido`：

  ```bash
  pip install kaleido
  ```
* 可根据需求增减指标、图表或多策略对比

---

*Happy Trading & Analysis!*
