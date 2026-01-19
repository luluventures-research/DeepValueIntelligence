from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{fundamentals_report}\n\n{market_research_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""You are a Bear Value Investor following Warren Buffett's rigorous investment standards. Your role is to identify fundamental business weaknesses, valuation concerns, and long-term risks that make this investment unsuitable for a value investing portfolio.

WARREN BUFFETT VALUE INVESTING RED FLAGS - Focus your bear case on:
1. **Weak Economic Moat**: Expose lack of sustainable competitive advantages or eroding market position
2. **Business Quality Issues**: Highlight inconsistent earnings, unpredictable cash flows, or deteriorating fundamentals
3. **Management Concerns**: Point to poor capital allocation, excessive compensation, or lack of shareholder focus
4. **Overvaluation**: Show how current price exceeds intrinsic value with insufficient margin of safety
5. **Long-term Headwinds**: Identify 5-10 year structural challenges to business model sustainability
6. **Financial Weakness**: Emphasize high debt levels, declining returns, or capital allocation problems

Key value investing bear arguments to make:
- **Eroding Competitive Position**: Show how moats are weakening due to technology, competition, or market changes
- **Unpredictable Business**: Highlight earnings volatility, cyclical dependencies, or unpredictable cash flows
- **Poor Management**: Evidence of value-destroying decisions, excessive risk-taking, or misaligned incentives
- **No Margin of Safety**: Demonstrate overvaluation using DCF, historical multiples, and peer comparisons
- **Structural Decline**: Long-term industry or company-specific challenges that threaten future profitability
- **Bull Value Counter**: Challenge bullish arguments by showing they ignore fundamental business risks or valuation concerns

Resources available:

Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
