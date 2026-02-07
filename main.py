import argparse
from datetime import date as _date, datetime
from pathlib import Path

from investingagents.graph.trading_graph import InvestingAgentsGraph
from investingagents.default_config import DEFAULT_CONFIG

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run InvestingAgents analysis.")
    parser.add_argument("--ticker", default="NVDA", help="Stock ticker symbol.")
    parser.add_argument(
        "--date",
        default=_date.today().isoformat(),
        help="Trade date in YYYY-MM-DD format (defaults to today).",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google", "anthropic"],
        default="google",
        help="LLM provider to use.",
    )
    parser.add_argument("--deep-model", default="gemini-3-pro-preview", help="Deep thinking model name.")
    parser.add_argument("--quick-model", default="gemini-3-flash-preview", help="Quick thinking model name.")
    parser.add_argument("--backend-url", default="https://generativelanguage.googleapis.com/v1", help="LLM backend URL.")
    parser.add_argument("--max-debate-rounds", type=int, default=1, help="Number of debate rounds.")
    parser.add_argument(
        "--online-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable online tools (default: enabled).",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "google", "ollama", "none"],
        default="",
        help="Embedding provider (default: auto).",
    )
    parser.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model name (provider-specific).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    return parser


def _ensure_string(content) -> str:
    """Convert content to string, extracting only 'text' field from Gemini's format."""
    if content is None:
        return ""
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                import ast
                parsed = ast.literal_eval(content)
                return _ensure_string(parsed)
            except (ValueError, SyntaxError):
                pass
        return content
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return content["text"]
        return str(content)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        return "\n\n".join(text_parts)
    return str(content)


def _filter_trading_recommendations(content: str) -> str:
    """
    Filter out explicit BUY/SELL/HOLD recommendations from content for the Intelligence Summary.
    Removes lines containing explicit trading decisions while preserving the analysis.
    """
    import re

    if not content:
        return content

    lines = content.split("\n")
    filtered_lines = []

    skip_patterns = [
        r"^\s*\*{0,2}(FINAL\s+)?(DECISION|RECOMMENDATION|VERDICT)\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD|STRONG\s+BUY|STRONG\s+SELL)",
        r"^\s*\*{0,2}(The\s+)?(recommendation|decision)\s+(is\s+to\s+)?(BUY|SELL|HOLD)",
        r"^\s*\*{0,2}(TRANSACTION\s+PROPOSAL)\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)",
        r"^\s*-?\s*\*{0,2}Action\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)",
        r"^\s*>\s*\*{0,2}(DECISION|RECOMMENDATION)\s*:\s*(BUY|SELL|HOLD)",
    ]

    for line in lines:
        skip_line = False
        for pattern in skip_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                skip_line = True
                break
        if not skip_line:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _generate_comprehensive_report(final_state, ticker, analysis_date) -> str:
    """Generate a comprehensive markdown report matching the CLI format."""
    report_lines = []

    # Header
    report_lines.append(f"# {ticker} Deep Value Intelligence")
    report_lines.append("")
    report_lines.append(f"**Analysis Date:** {analysis_date}")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Table of Contents
    report_lines.append("## Table of Contents")
    report_lines.append("")
    report_lines.append("1. [Intelligence Summary](#intelligence-summary)")
    report_lines.append("2. [Analyst Team Reports](#analyst-team-reports)")
    report_lines.append("   - [Fundamentals Analysis](#fundamentals-analysis)")
    report_lines.append("   - [Value Analysis (Buffett)](#value-analysis-buffett)")
    report_lines.append("   - [Growth Analysis (Lynch/Druckenmiller/Fisher)](#growth-analysis-lynchdruckenmillerfisher)")
    report_lines.append("   - [Market Analysis](#market-analysis)")
    report_lines.append("   - [Social Sentiment Analysis](#social-sentiment-analysis)")
    report_lines.append("   - [News Analysis](#news-analysis)")
    report_lines.append("3. [Research Team Decision](#research-team-decision)")
    report_lines.append("4. [Trading Team Plan](#trading-team-plan)")
    report_lines.append("5. [Deep Values Strategy](#deep-values-strategy)")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Intelligence Summary
    report_lines.append("## Intelligence Summary")
    report_lines.append("")
    if final_state.get("final_trade_decision"):
        decision_content = _ensure_string(final_state["final_trade_decision"])
        filtered_content = _filter_trading_recommendations(decision_content)
        report_lines.append(filtered_content)
    else:
        report_lines.append("*No executive summary available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Analyst Team Reports
    report_lines.append("## Analyst Team Reports")
    report_lines.append("")

    report_lines.append("### Fundamentals Analysis")
    report_lines.append("")
    if final_state.get("fundamentals_report"):
        report_lines.append(_ensure_string(final_state["fundamentals_report"]))
    else:
        report_lines.append("*No fundamentals analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Value Analysis (Buffett)")
    report_lines.append("")
    report_lines.append("> *Following Warren Buffett's value investing philosophy: Circle of Competence, Economic Moat, Management Quality, and Margin of Safety.*")
    report_lines.append("")
    if final_state.get("value_report"):
        report_lines.append(_ensure_string(final_state["value_report"]))
    else:
        report_lines.append("*No value analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Growth Analysis (Lynch/Druckenmiller/Fisher)")
    report_lines.append("")
    report_lines.append("> *Following Peter Lynch's GARP, Stanley Druckenmiller's macro-aware approach, and Philip Fisher's scuttlebutt methodology.*")
    report_lines.append("")
    if final_state.get("growth_report"):
        report_lines.append(_ensure_string(final_state["growth_report"]))
    else:
        report_lines.append("*No growth analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Market Analysis")
    report_lines.append("")
    if final_state.get("market_report"):
        report_lines.append(_ensure_string(final_state["market_report"]))
    else:
        report_lines.append("*No market analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### Social Sentiment Analysis")
    report_lines.append("")
    if final_state.get("sentiment_report"):
        report_lines.append(_ensure_string(final_state["sentiment_report"]))
    else:
        report_lines.append("*No sentiment analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    report_lines.append("### News Analysis")
    report_lines.append("")
    if final_state.get("news_report"):
        report_lines.append(_ensure_string(final_state["news_report"]))
    else:
        report_lines.append("*No news analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Research Team Decision
    report_lines.append("## Research Team Decision")
    report_lines.append("")
    report_lines.append("> *The Research Manager synthesizes the Bull vs Bear debate to provide an AI strategy.*")
    report_lines.append("")
    if final_state.get("investment_plan"):
        report_lines.append(_ensure_string(final_state["investment_plan"]))
    else:
        report_lines.append("*No research team decision available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Trading Team Plan
    report_lines.append("## Trading Team Plan")
    report_lines.append("")
    report_lines.append("> *The Trader creates a specific investment plan based on the research team's analysis.*")
    report_lines.append("")
    if final_state.get("trader_investment_plan"):
        report_lines.append(_ensure_string(final_state["trader_investment_plan"]))
    else:
        report_lines.append("*No trading plan available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Deep Values Strategy
    report_lines.append("## Deep Values Strategy")
    report_lines.append("")
    report_lines.append("> *The Risk Management Team (Aggressive, Conservative, Neutral) debates and the Portfolio Manager provides the final strategy.*")
    report_lines.append("")
    if final_state.get("final_trade_decision"):
        report_lines.append(_ensure_string(final_state["final_trade_decision"]))
    else:
        report_lines.append("*No AI investing strategy available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Footer
    report_lines.append("## Disclaimer")
    report_lines.append("")
    report_lines.append("*This analysis is generated by Deep Value Intelligence, an AI-powered multi-agent trading framework. ")
    report_lines.append("This report is for informational and educational purposes only and does not constitute financial advice. ")
    report_lines.append("Always conduct your own research and consult with qualified financial advisors before making investment decisions.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append(f"*Report generated by Deep Value Intelligence on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(report_lines)


def main() -> None:
    args = _build_parser().parse_args()

    # Create a custom config
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.provider
    config["backend_url"] = args.backend_url
    config["deep_think_llm"] = args.deep_model
    config["quick_think_llm"] = args.quick_model
    config["max_debate_rounds"] = args.max_debate_rounds
    config["online_tools"] = args.online_tools
    if args.embedding_provider:
        config["embedding_provider"] = args.embedding_provider
    if args.embedding_model:
        config["embedding_model"] = args.embedding_model

    # Initialize with custom config
    ta = InvestingAgentsGraph(debug=args.debug, config=config)

    # forward propagate
    final_state, decision = ta.propagate(args.ticker, args.date)

    print(decision)

    # Save the final markdown report to the standard log folder (CLI format)
    report_md = _generate_comprehensive_report(final_state, args.ticker, args.date)
    results_dir = Path(config["results_dir"]) / args.ticker / args.date
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{args.ticker}_deep_value_intelligence_{args.date}.md"
    report_path.write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    main()

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
