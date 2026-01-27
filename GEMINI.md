## Project Overview

This project, `TradingAgents`, is a multi-agent LLM financial trading framework. It uses a team of specialized AI agents to analyze financial markets, debate potential trading strategies, and make decisions. The framework is designed to mirror the structure of a real-world trading firm, with roles for analysts, researchers, traders, and risk managers.

The system is built on Python and heavily utilizes the `LangGraph` library to create and manage the agentic workflows. It supports various LLM backends including OpenAI, Google Gemini, and Anthropic, with a default configuration pointing to a local Ollama setup.

The core of the application is a `TradingAgentsGraph` which orchestrates the interactions between the different agents. The process flows from data gathering and analysis by a team of analysts (Market, Social, News, Fundamentals), through a debate phase by a research team (Bull vs. Bear), to a final decision-making process involving a trader and a risk management team.

## Key Components

*   **`TradingAgentsGraph` (`tradingagents/graph/trading_graph.py`):** The central class that defines and runs the entire multi-agent workflow using `LangGraph`.
*   **Agents (`tradingagents/agents/`):** Individual agents with specialized roles, such as `FundamentalsAnalyst`, `MarketAnalyst`, `BullResearcher`, `Trader`, etc.
*   **Tools (`tradingagents/dataflows/`):** A collection of tools for gathering financial data from various sources like Yahoo Finance, Reddit, Finnhub, and Google News.
*   **CLI (`cli/main.py`):** A rich, interactive command-line interface built with `Typer` and `rich` for running the analysis and visualizing the process in real-time.
*   **Configuration (`tradingagents/default_config.py`):** A centralized configuration file for managing LLM providers, API keys, file paths, and other parameters.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/luluventures-research/TradingAgents.git
    cd TradingAgents
    ```

2.  **Create a virtual environment:**
    The project recommends using `conda`.
    ```bash
    conda create -n tradingagents python=3.13
    conda activate tradingagents
    ```

3.  **Install dependencies:**
    The project uses `pip` to manage dependencies listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    The application requires API keys for the selected LLM provider and for financial data. These should be set as environment variables.

    *   **For OpenAI:**
        ```bash
        export OPENAI_API_KEY="your-openai-api-key"
        ```
    *   **For Google:**
        ```bash
        export GOOGLE_API_KEY="your-google-api-key"
        ```
    *   **For Finnhub (Financial Data):**
        ```bash
        export FINNHUB_API_KEY="your-finnhub-api-key"
        ```

## Running the Application

There are two main ways to run the analysis.

### 1. Interactive CLI

The recommended way to run the application is through its interactive CLI, which provides a rich visualization of the agents' progress.

```bash
python -m cli.main
```

This will guide you through selecting a stock ticker, analysis date, LLM models, and other settings before starting the analysis.

### 2. Direct Python Script

For development or direct execution, you can run `main.py`. This script is pre-configured to run an analysis on "NVDA" and can be easily modified.

```bash
python main.py
```

## Configuration

The main configuration is located in `tradingagents/default_config.py`. You can create a `local_config.py` to override these settings without modifying the original files.

Key configuration options include:

*   `llm_provider`: "openai", "google", or "anthropic".
*   `deep_think_llm`: The model name for more complex reasoning tasks.
*   `quick_think_llm`: The model name for faster, less complex tasks.
*   `backend_url`: The API endpoint for the LLM provider.
*   `max_debate_rounds`: The number of debate rounds for the research team.
*   `online_tools`: A boolean to enable or disable tools that require internet access.

The default configuration is set up for a local Ollama instance. To use other providers like OpenAI or Google, you need to set the appropriate environment variables and modify the configuration or select the provider in the interactive CLI.

## Development Conventions

*   **Dependency Management:** Dependencies are managed in `requirements.txt` and `pyproject.toml`.
*   **Code Style:** The code appears to follow standard Python conventions. There is no explicit linter configuration file found, but the code is well-structured and readable.
*   **Modularity:** The project is highly modular, with clear separation between agents, tools, and the main graph orchestration. This makes it easy to extend and modify.
*   **Testing:** No dedicated test files were found in the initial analysis, but the modular structure would lend itself to unit testing of individual agents and tools.
*   **Entry Points:** The project has two main entry points: `cli/main.py` for the interactive CLI and `main.py` for direct script execution.
