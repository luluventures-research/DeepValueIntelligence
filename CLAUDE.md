# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InvestingAgents is a multi-agent LLM financial trading framework that mirrors real-world trading firm dynamics. Specialized LLM-powered agents (analysts, researchers, traders, risk managers) collaborate to evaluate market conditions and make trading decisions through structured debates.

## Common Commands

```bash
# Run the interactive CLI (recommended)
python -m cli.main

# Run direct Python script for development
python main.py

# Install dependencies
pip install -r requirements.txt
```

## Required Environment Variables

```bash
export FINNHUB_API_KEY="your-finnhub-api-key"    # Required for financial data
export OPENAI_API_KEY="your-openai-api-key"      # For OpenAI models
export GOOGLE_API_KEY="your-google-api-key"      # For Gemini models
export ANTHROPIC_API_KEY="your-anthropic-api-key" # For Claude models
```

## Architecture

### Core Flow
The system processes trading decisions through a LangGraph-based pipeline:

```
Analyst Team → Research Team (Bull/Bear Debate) → Trader → Risk Management Team → Portfolio Manager
```

### Key Components

**`investingagents/graph/trading_graph.py`** - `InvestingAgentsGraph` is the central orchestrator that:
- Initializes LLMs based on provider config (OpenAI, Google, Anthropic)
- Creates tool nodes for different data sources (market, social, news, fundamentals)
- Manages the multi-agent workflow via LangGraph
- Entry point: `propagate(ticker, date)` returns final state and trading decision

**`investingagents/agents/`** - Specialized agents organized by role:
- `analysts/` - Market, News, Social Media, Fundamentals analysts
- `researchers/` - Bull and Bear researchers for debate
- `managers/` - Research Manager and Risk Manager
- `risk_mgmt/` - Aggressive, Conservative, Neutral debators
- `trader/` - Makes trading decisions based on research

**`investingagents/dataflows/interface.py`** - Data retrieval tools including:
- Yahoo Finance, SimFin (balance sheets, cash flow, income statements)
- Finnhub (news, insider sentiment, transactions)
- Reddit, Google News for sentiment
- Technical indicators via stockstats

**`investingagents/default_config.py`** - Configuration for LLM providers, debate rounds, data paths

**`cli/main.py`** - Rich interactive CLI with real-time progress visualization

### Agent States
Defined in `investingagents/agents/utils/agent_states.py`:
- `AgentState` - Main workflow state
- `InvestDebateState` - Bull vs Bear research debate
- `RiskDebateState` - Risk management discussion

### Configuration

Key config options in `DEFAULT_CONFIG`:
- `llm_provider`: "openai", "google", or "anthropic"
- `deep_think_llm` / `quick_think_llm`: Model names for reasoning tasks
- `max_debate_rounds`: Number of research debate iterations
- `online_tools`: Enable real-time data (vs cached data)

Model selection is automatic based on model name prefix (gemini/google → Google, claude → Anthropic, else → OpenAI).

## Python Usage

```python
from investingagents.graph.trading_graph import InvestingAgentsGraph
from investingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o"
config["quick_think_llm"] = "gpt-4o-mini"

ta = InvestingAgentsGraph(debug=True, config=config)
final_state, decision = ta.propagate("NVDA", "2024-05-10")
```

## Data Sources

- **Online mode** (`online_tools=True`): Real-time data via APIs
- **Offline mode**: Uses cached data from `data_dir` path (configured for backtesting with "Lulu TradingDB")

Results are saved to `results/{ticker}/{date}/reports/` directory.
