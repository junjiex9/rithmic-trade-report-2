# 自动化交易分析报告生成器

一个基于 Streamlit 的可视化交易报告工具，支持：
- 数据清洗：解析 Rithmic/ATAS 导出 CSV
- 交互式图表：累积盈亏、日/小时盈亏、持仓时长、Monte Carlo 模拟、滑点分布、舆情热力图
- 核心指标：夏普比率、胜率、盈亏比、最大回撤、Calmar 比率等
- 导出功能：Excel (.xlsx) 与 PDF 报告
- 自定义设置：缓存、快照保留、回撤回溯期

## 部署

1. 在 [Streamlit Cloud](https://streamlit.io/cloud) 创建应用。  
2. 将代码文件 `streamlit_app.py`、`requirements.txt`、`README.md` 上传至仓库根目录。  
3. 在 Cloud 中设置主运行文件为 `streamlit_app.py`。  
4. 点击 Deploy，即可在线访问报告生成器。

## 本地运行

```bash
git clone <repo_url>
cd <repo>
pip install -r requirements.txt
streamlit run streamlit_app.py
