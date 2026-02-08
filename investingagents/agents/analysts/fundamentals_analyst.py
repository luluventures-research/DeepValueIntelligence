from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
import time
import json
from investingagents.dataflows.interface import get_enhanced_fundamentals


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            # Determine which fundamentals tool to use based on model configuration
            deep_model = toolkit.config.get("deep_think_llm", "")
            quick_model = toolkit.config.get("quick_think_llm", "")
            
            using_gemini = (
                deep_model.startswith(("gemini", "google")) or 
                quick_model.startswith(("gemini", "google"))
            )
            
            if using_gemini:
                tools = [toolkit.get_enhanced_fundamentals, toolkit.get_fundamentals_google]
            else:
                tools = [toolkit.get_enhanced_fundamentals, toolkit.get_fundamentals_openai]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "You are a quantitative fundamental analyst specializing in Warren Buffett's value investing methodology. "
            "Conduct a comprehensive 10-year historical analysis of the company's financial metrics.\n\n"
            
            "REQUIRED ANALYSIS - Collect and analyze the following metrics for the PAST 10 YEARS:\n"
            "1. Market Metrics: Market Price, Total Market Cap, P/E Ratio, P/B Ratio\n"
            "2. Profitability: Return on Equity (ROE), Return on Invested Capital (ROIC), EPS, Revenue, Gross Profit, Operating Margin, Net Income, Net Margin, Net Margin Gain\n"
            "3. Growth Metrics: Revenue Growth Rate, Net Income Growth Rate, Free Cash Flow Growth Rate, Cash Flow for Owner Growth Rate\n"
            "4. Balance Sheet: Total Book Value, Total Assets, Total Debt, Debt-to-Equity Ratio, Debt-to-Asset Ratio, Cash/Cash Equivalents, Shareholder's Equity\n"
            "5. Cash Flow: Free Cash Flow, Dividends per Share\n\n"
            
            "DATA VALIDATION: Cross-validate financial data from multiple trusted sources when available.\n\n"
            
            "COMPARATIVE ANALYSIS: For each metric, compare current values with 10-year averages and explain:\n"
            "- Current standing relative to historical performance\n"
            "- Trends and patterns over the decade\n"
            "- Significance of deviations from historical norms\n\n"
            
            "WARREN BUFFETT VALUE INVESTING ANALYSIS:\n"
            "- Economic Moat: Assess competitive advantages and business durability\n"
            "- Financial Strength: Analyze debt levels, cash position, and financial stability\n"
            "- Predictable Earnings: Evaluate consistency and reliability of earnings over 10 years\n"
            "- Management Performance: ROE trends, capital allocation efficiency, dividend policy\n"
            "- Value Assessment: Compare current valuation to historical averages and intrinsic value\n"
            "- Quality of Business: Revenue predictability, margin stability, competitive position\n\n"
            
            "DISCOUNTED CASH FLOW (DCF) ANALYSIS - Calculate fair value using three scenarios:\n"
            "1. CONSERVATIVE: Use the LOWEST free cash flow growth rate from the past 10 years\n"
            "2. AVERAGE: Use the AVERAGE free cash flow growth rate from the past 10 years\n" 
            "3. OPTIMISTIC: Use the HIGHEST free cash flow growth rate from the past 10 years\n"
            "For each scenario, use a 10% discount rate and 2.5% terminal growth rate. Show detailed calculations.\n\n"
            
            "DELIVERABLES:\n"
            "Before you start writing the final report, make a list of all the metrics you have gathered. Make sure you have all the required metrics before you proceed.\n"
            "To ensure all metrics are included, you can use a JSON object to structure the data before generating the final table.\n"
            "- 10-year comprehensive historical data table with all required metrics\n"
            "- Warren Buffett-style qualitative analysis with specific insights\n"
            "- Three-scenario DCF valuation with detailed calculations and fair value ranges\n"
            "- Current vs historical average comparison table\n"
            "- Key investment insights, red flags, and opportunities\n"
            "- Final investment recommendation with supporting rationale\n"
            "Finally, double-check your work to ensure all the required metrics are present in the final table.\n"
            "Make sure to include as much detail as possible. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions.\n"
            "Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        report = ""
        try:
            result = chain.invoke(state["messages"])
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}".lower()
            if "timeout" not in error_text:
                raise

            # Fallback path so one LLM timeout does not abort the entire graph.
            try:
                fallback_data = get_enhanced_fundamentals(ticker, current_date)
                report = (
                    "### Fundamentals Analyst Fallback Report\n"
                    f"Primary LLM analysis timed out for {ticker} on {current_date}. "
                    "Using direct data retrieval fallback so the workflow can continue.\n\n"
                    f"{fallback_data}\n\n"
                    "Note: For full narrative analysis, rerun with a higher timeout via "
                    "`--llm-timeout 3600` or use a faster model."
                )
            except Exception as fallback_exc:
                report = (
                    "### Fundamentals Analyst Fallback Report\n"
                    f"Primary LLM analysis timed out for {ticker} on {current_date}, "
                    "and fallback data retrieval also failed.\n\n"
                    f"Fallback error: {type(fallback_exc).__name__}: {fallback_exc}\n\n"
                    "Consider rerunning with a higher timeout via `--llm-timeout 3600`."
                )

            result = AIMessage(content=report)

        if len(getattr(result, "tool_calls", [])) == 0 and not report:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
