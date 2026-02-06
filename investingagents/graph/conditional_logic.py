# InvestingAgents/graph/conditional_logic.py

import logging

from investingagents.agents.utils.agent_states import AgentState

logger = logging.getLogger(__name__)


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    # Absolute safety limits to prevent infinite loops
    ABSOLUTE_MAX_DEBATE_COUNT = 10
    ABSOLUTE_MAX_RISK_COUNT = 15

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_value(self, state: AgentState):
        """Determine if value analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_value"
        return "Msg Clear Value"

    def should_continue_growth(self, state: AgentState):
        """Determine if growth analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_growth"
        return "Msg Clear Growth"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""
        count = state["investment_debate_state"]["count"]

        # Safety check: absolute limit to prevent infinite loops
        if count >= self.ABSOLUTE_MAX_DEBATE_COUNT:
            logger.warning(
                f"Debate hit absolute safety limit! count={count}, limit={self.ABSOLUTE_MAX_DEBATE_COUNT}. "
                "Forcing termination to Research Manager."
            )
            return "Research Manager"

        if count >= 2 * self.max_debate_rounds:
            logger.debug(f"Debate complete after {count} rounds, moving to Research Manager")
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        count = state["risk_debate_state"]["count"]

        # Safety check: absolute limit to prevent infinite loops
        if count >= self.ABSOLUTE_MAX_RISK_COUNT:
            logger.warning(
                f"Risk analysis hit absolute safety limit! count={count}, limit={self.ABSOLUTE_MAX_RISK_COUNT}. "
                "Forcing termination to Risk Judge."
            )
            return "Risk Judge"

        if count >= 3 * self.max_risk_discuss_rounds:
            logger.debug(f"Risk analysis complete after {count} rounds, moving to Risk Judge")
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"
        return "Risky Analyst"
