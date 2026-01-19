from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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

        prompt = f"""You are a Bull Value Investor following Warren Buffett's investment philosophy. Your task is to build a strong, evidence-based case for long-term value creation, emphasizing business fundamentals, competitive moats, and intrinsic value opportunities.

WARREN BUFFETT VALUE INVESTING PRINCIPLES - Focus your bull case on:
1. **Economic Moat**: Demonstrate the company's durable competitive advantages and barriers to entry
2. **Business Quality**: Emphasize consistent earnings, predictable cash flows, and strong fundamentals
3. **Management Excellence**: Highlight effective capital allocation and shareholder-oriented leadership
4. **Intrinsic Value**: Show how current market price offers significant margin of safety below fair value
5. **Long-term Growth**: Focus on sustainable 10+ year business prospects, not short-term market movements
6. **Financial Strength**: Emphasize strong balance sheet, low debt, and high returns on equity/capital

Key value investing arguments to make:
- **Sustainable Competitive Advantages**: Highlight moats like brand power, network effects, cost advantages, or regulatory barriers
- **Predictable Business Model**: Emphasize revenue/earnings consistency and cash flow reliability over time
- **Quality Management**: Show evidence of smart capital allocation, shareholder returns, and strategic vision
- **Undervaluation**: Present DCF analysis, P/E comparisons, and margin of safety calculations
- **Long-term Perspective**: Counter short-term concerns with 5-10 year business fundamentals view
- **Bear Value Counter**: Address bear concerns through the lens of business quality and long-term value creation

Resources available:
Market research report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
Company fundamentals report: {fundamentals_report}
Conversation history of the debate: {history}
Last bear argument: {current_response}
Reflections from similar situations and lessons learned: {past_memory_str}
Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past.
"""

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
