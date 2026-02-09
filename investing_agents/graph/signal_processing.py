# InvestingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes analysis text to extract an investment stance label."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full analysis output to extract the core stance.

        Args:
            full_signal: Complete analysis text

        Returns:
            Extracted stance (ADVOCATE, WATCH, or AVOID)
        """
        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze financial reports. Your task is to extract the investment stance: ADVOCATE, WATCH, or AVOID. Provide only the extracted stance (ADVOCATE, WATCH, or AVOID) as your output, without adding any additional text or information.",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content
