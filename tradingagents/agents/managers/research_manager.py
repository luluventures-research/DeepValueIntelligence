import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{fundamentals_report}\n\n{market_research_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the value investing portfolio manager following Warren Buffett's philosophy, your role is to critically evaluate this round of debate and make a definitive decision: align with the bear analyst, the bull analyst, or choose Hold only if it is strongly justified based on the arguments presented.

WARREN BUFFETT VALUE INVESTING FRAMEWORK - Apply these principles when evaluating the debate:
1. **Business Quality First**: Prioritize companies with durable competitive advantages (economic moats)
2. **Long-term Perspective**: Focus on 10+ year business prospects, not short-term market fluctuations
3. **Intrinsic Value**: Buy only when market price is significantly below intrinsic value (margin of safety)
4. **Management Quality**: Assess capital allocation skills and shareholder-oriented management
5. **Predictable Earnings**: Favor businesses with consistent, predictable cash flows
6. **Financial Strength**: Strong balance sheets with minimal debt and high returns on equity

SELECTIVE CONTRARIAN ANALYSIS - Apply critical thinking to news and sentiment:
- **News Deep Dive**: Look beyond headlines. What do recent news events really mean for the business's long-term competitive position and cash flows?
- **Sentiment Contrarianism**: When is fear creating opportunity? When is euphoria masking risk? What is the crowd missing?
- **Market Psychology**: Analyze what current sentiment reveals about potential mispricings. Are temporary factors creating permanent value destruction in perception?
- **Critical Questions**: What assumptions are markets making? What would have to be true for current sentiment to be justified? What evidence contradicts popular narratives?
- **Selective Approach**: Be contrarian when you have superior business insight, not just to oppose consensus. Sometimes markets are efficient.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning through the lens of value investing principles. Your recommendation—Buy, Sell, or Hold—must be clear and actionable. Avoid defaulting to Hold simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments about business fundamentals and long-term value creation.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments about business quality and intrinsic value.
Rationale: An explanation of why these arguments lead to your conclusion from a value investing perspective.
Strategic Actions: Concrete steps for implementing the recommendation with a long-term focus.

Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting.

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Here is the debate:
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
