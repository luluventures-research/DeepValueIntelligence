import argparse
from datetime import date as _date

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
    _, decision = ta.propagate(args.ticker, args.date)
    print(decision)


if __name__ == "__main__":
    main()

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
