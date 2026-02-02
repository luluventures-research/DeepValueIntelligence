from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


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
            "- 10-year comprehensive historical data table with all required metrics\n"
            "- Warren Buffett-style qualitative analysis with specific insights\n"
            "- Three-scenario DCF valuation with detailed calculations and fair value ranges\n"
            "- Current vs historical average comparison table\n"
            "- Key investment insights, red flags, and opportunities\n"
            "- Final investment recommendation with supporting rationale\n"
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

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
        else:
            # If there are tool calls, we'll let the tools execute and come back
            # The report will be generated on the next iteration when tools are done
            pass

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
