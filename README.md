<p align="center">
  <img src="assets/deep_value_intelligence.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="./assets/lulu_research.png" target="_blank"><img alt="WeChat" src="https://img.shields.io/badge/WeChat-LuluResearch-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://x.com/LuluResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-LuluResearch-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="https://github.com/luluventures-research/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/Join_GitHub_Community-luluventures-research-14C290?logo=discourse"/></a>
</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/luluventures-research/DeepValueIntelligence?lang=zh">中文</a>
</div>

---

# Deep Value Intelligence: Multi-Agents LLM Financial Framework for Value Investors 

> 🎉 **Deep Value Intelligence** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

<div align="center">

🚀 [DeepValueIntelligence](#deep-value-intelligence-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#deepvalueintelligence-package) | 🤝 [Contributing](#contributing)

</div>

## Deep Value Intelligence Framework

DeepValueIntelligence is a multi-agent investing framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs investing decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> DeepValueIntelligence framework is designed for research purposes. Investing performance may vary based on many factors, including the chosen backbone language models, model temperature, investing periods, the quality of data, and other non-deterministic factors. [disclaimer](https://luluvc.com/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles. This ensures the system achieves a robust, scalable approach to market analysis and decision-making.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics through a 10-year comprehensive historical analysis, identifying intrinsic values and potential red flags. The analysis utilizes an improved prompt to ensure completeness and accuracy of all required metrics.
- Sentiment Analyst: Analyzes social media and public sentiment using sentiment scoring algorithms to gauge short-term market mood.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Investing Agent
- Composes reports from the analysts and researchers to make informed investing decisions. It determines the timing and magnitude of trades based on comprehensive market insights.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts investing strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone DeepValueIntelligence:
```bash
git clone https://github.com/luluventures-research/DeepValueIntelligence.git
cd DeepValueIntelligence
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n DeepValueIntelligence python=3.13
conda activate DeepValueIntelligence
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Required APIs

You will also need the FinnHub API for financial data. All of our code is implemented with the free tier.
```bash
export FINNHUB_API_KEY=$YOUR_FINNHUB_API_KEY
```

You will need the OpenAI API for all the agents.
```bash
export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
```

### CLI Usage

You can also try out the CLI directly by running:
```bash
python -m cli.main
```
You will see a screen where you can select your desired tickers, date, LLMs, research depth, etc.

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Online Tools vs Offline

The framework can run in two modes:
- **Online tools enabled**: agents fetch live data (Yahoo Finance, Finnhub, Google News, Reddit, stockstats).
- **Online tools disabled**: agents rely on cached/offline sources (e.g., SimFin data and cached reports). This is useful for deterministic backtests or running without API keys.

## DeepValueIntelligence Package

### Implementation Details

We built DeepValueIntelligence with LangGraph to ensure flexibility and modularity. We utilize `gemini-3-pro-preview` as our deep thinking LLM and `gemini-3-flash-preview` as our fast thinking LLM for our experiments. The default LLM provider is Google, and the backend URL is `https://generativelanguage.googleapis.com/v1`. However, for testing purposes, we recommend you use more cost-effective models as our framework makes **lots of** API calls.

### Python Usage

To use DeepValueIntelligence inside your code, you can import the `investing_agents` module and initialize a `InvestingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from investing_agents.graph.trading_graph import InvestingAgentsGraph
from investing_agents.default_config import DEFAULT_CONFIG

ta = InvestingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from investing_agents.graph.trading_graph import InvestingAgentsGraph
from investing_agents.default_config import DEFAULT_CONFIG

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4.1-nano"  # Use a different model
config["quick_think_llm"] = "gpt-4.1-nano"  # Use a different model
config["max_debate_rounds"] = 1  # Increase debate rounds
config["online_tools"] = True # Use online tools or cached data

# Initialize with custom config
ta = InvestingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

> For `online_tools`, we recommend enabling them for experimentation, as they provide access to real-time data. The agents' offline tools rely on cached data from our **Lulu TradingDB**, a curated dataset we use for backtesting. We're currently in the process of refining this dataset, and we plan to release it soon alongside our upcoming projects. Stay tuned!

You can view the full list of configurations in `investing_agents/default_config.py`.

## Contributing

We welcome contributions from the community! Whether it's fixing a bug, improving documentation, or suggesting a new feature, your input helps make this project better.
