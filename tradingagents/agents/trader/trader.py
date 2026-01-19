import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{fundamentals_report}\n\n{market_research_report}\n\n{sentiment_report}\n\n{news_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. This plan incorporates insights from fundamental business analysis, market trends, macroeconomic indicators, and sentiment analysis. Use this plan as a foundation for evaluating your long-term investment decision.\n\nProposed Investment Plan: {investment_plan}\n\nApply selective contrarian thinking to critically analyze what the news and sentiment reveal about potential opportunities or risks. Make an informed and strategic investment decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a Warren Buffett-style value investor focused on long-term wealth creation through business ownership. Your investment decisions are based on fundamental business analysis, not market speculation or short-term trading.

WARREN BUFFETT VALUE INVESTING FRAMEWORK:
1. **Buy Businesses, Not Stocks**: Focus on owning pieces of great businesses for decades
2. **Circle of Competence**: Only invest in businesses you can understand completely
3. **Margin of Safety**: Buy only when intrinsic value significantly exceeds market price
4. **Quality Management**: Invest with managers who allocate capital wisely and treat shareholders as partners
5. **Economic Moats**: Prioritize companies with durable competitive advantages
6. **Long-Term Perspective**: Think like a business owner holding for 10+ years, ignore short-term market noise

SELECTIVE CONTRARIAN THINKING:
- **News Analysis**: Look beyond headlines to understand underlying business implications. Ask: "What does this really mean for the business 5-10 years from now?"
- **Sentiment Contrarianism**: When crowds are fearful about quality businesses, consider it an opportunity. When euphoric about mediocre businesses, exercise caution.
- **Critical Questions**: What are markets missing? What temporary factors are distorting perceptions? Is current sentiment rational given business fundamentals?
- **Selective Approach**: Be contrarian when you have superior business insight, not just to be different. Sometimes the crowd is right.

DECISION CRITERIA (in priority order):
- Business quality and predictable earnings power
- Strength and sustainability of competitive advantages (moats)
- Management quality and capital allocation track record
- Intrinsic value vs current market price (margin of safety requirement)
- Long-term industry dynamics and growth prospects
- Balance sheet strength and financial stability

INVESTMENT PHILOSOPHY:
- Hold quality businesses through market volatility
- Buy more when great businesses become cheaper due to temporary sentiment
- Sell only when: business fundamentals deteriorate, better opportunities arise, or price exceeds intrinsic value
- Ignore short-term market movements and sentiment swings while analyzing what they reveal about opportunity
- Focus on what the business will earn over the next 5-10 years

Based on your analysis, provide a specific recommendation following value investing principles with selective contrarian insights. End with a firm decision and always conclude your response with 'FINAL INVESTMENT PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. 

Learn from past value investing decisions and mistakes: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
