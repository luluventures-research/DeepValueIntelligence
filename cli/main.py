from typing import Optional
import datetime
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.live import Live
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.models import AnalystType
from cli.utils import *

console = Console()

app = typer.Typer(
    name="DeepValueIntelligence",
    help="Deep Value Intelligence CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


def _ensure_string(content):
    """Convert content to string, extracting only 'text' field from Gemini's format."""
    if content is None:
        return ""
    if isinstance(content, str):
        # Check if it's a string representation of a dict/list
        stripped = content.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                import ast
                parsed = ast.literal_eval(content)
                return _ensure_string(parsed)
            except (ValueError, SyntaxError):
                pass
        return content
    if isinstance(content, dict):
        # Handle single dict: {'type': 'text', 'text': '...', 'extras': {...}}
        if content.get('type') == 'text' and 'text' in content:
            return content['text']
        return str(content)
    if isinstance(content, list):
        # Handle Gemini's list format: [{'type': 'text', 'text': '...'}]
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        return '\n\n'.join(text_parts)
    return str(content)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {
            # Analyst Team
            "Fundamentals Analyst": "pending",
            "Value Analyst": "pending",
            "Growth Analyst": "pending",
            "Market Analyst": "pending",
            "Social Analyst": "pending",
            "News Analyst": "pending",
            # Research Team
            "Bull Researcher": "pending",
            "Bear Researcher": "pending",
            "Research Manager": "pending",
            # Trading Team
            "Trader": "pending",
            # Risk Management Team
            "Risky Analyst": "pending",
            "Neutral Analyst": "pending",
            "Safe Analyst": "pending",
            # Portfolio Management Team
            "Portfolio Manager": "pending",
        }
        self.current_agent = None
        self.report_sections = {
            "market_report": None,
            "sentiment_report": None,
            "news_report": None,
            "fundamentals_report": None,
            "value_report": None,
            "growth_report": None,
            "investment_plan": None,
            "trader_investment_plan": None,
            "final_trade_decision": None,
        }

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "value_report": "Value Analysis (Buffett)",
                "growth_report": "Growth Analysis (Lynch/Druckenmiller/Fisher)",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Deep Values Strategy",
            }
            # Ensure content is a string (handle Gemini list format)
            content_str = _ensure_string(latest_content)
            self.current_report = (
                f"### {section_titles[latest_section]}\n{content_str}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports
        if any(
            self.report_sections[section]
            for section in [
                "fundamentals_report",
                "value_report",
                "growth_report",
                "market_report",
                "sentiment_report",
                "news_report",
            ]
        ):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections["fundamentals_report"]:
                report_parts.append(
                    f"### Fundamentals Analysis\n{_ensure_string(self.report_sections['fundamentals_report'])}"
                )
            if self.report_sections["value_report"]:
                report_parts.append(
                    f"### Value Analysis (Buffett)\n{_ensure_string(self.report_sections['value_report'])}"
                )
            if self.report_sections["growth_report"]:
                report_parts.append(
                    f"### Growth Analysis (Lynch/Druckenmiller/Fisher)\n{_ensure_string(self.report_sections['growth_report'])}"
                )
            if self.report_sections["market_report"]:
                report_parts.append(
                    f"### Market Analysis\n{_ensure_string(self.report_sections['market_report'])}"
                )
            if self.report_sections["sentiment_report"]:
                report_parts.append(
                    f"### Social Sentiment\n{_ensure_string(self.report_sections['sentiment_report'])}"
                )
            if self.report_sections["news_report"]:
                report_parts.append(
                    f"### News Analysis\n{_ensure_string(self.report_sections['news_report'])}"
                )

        # Research Team Reports
        if self.report_sections["investment_plan"]:
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{_ensure_string(self.report_sections['investment_plan'])}")

        # Trading Team Reports
        if self.report_sections["trader_investment_plan"]:
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{_ensure_string(self.report_sections['trader_investment_plan'])}")

        # Deep Values Strategy
        if self.report_sections["final_trade_decision"]:
            report_parts.append("## Deep Values Strategy")
            report_parts.append(f"{_ensure_string(self.report_sections['final_trade_decision'])}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def update_display(layout, spinner_text=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to Deep Value Intelligence CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to Deep Value Intelligence",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team
    teams = {
        "Analyst Team": [
            "Fundamentals Analyst",
            "Value Analyst",
            "Growth Analyst",
            "Market Analyst",
            "Social Analyst",
            "News Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Risky Analyst", "Neutral Analyst", "Safe Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status[first_agent]
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status[agent]
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        # Truncate tool call args if too long
        if isinstance(args, str) and len(args) > 100:
            args = args[:97] + "..."
        all_messages.append((timestamp, "Tool", f"{tool_name}: {args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        # Convert content to string if it's not already
        content_str = content
        if isinstance(content, list):
            # Handle list of content blocks (Anthropic format)
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif item.get('type') == 'tool_use':
                        text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
                else:
                    text_parts.append(str(item))
            content_str = ' '.join(text_parts)
        elif not isinstance(content_str, str):
            content_str = str(content)
            
        # Truncate message content if too long
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp
    all_messages.sort(key=lambda x: x[0])

    # Calculate how many messages we can show based on available space
    # Start with a reasonable number and adjust based on content length
    max_messages = 12  # Increased from 8 to better fill the space

    # Get the last N messages that will fit in the panel
    recent_messages = all_messages[-max_messages:]

    # Add messages to table
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    if spinner_text:
        messages_table.add_row("", "Spinner", spinner_text)

    # Add a footer to indicate if messages were truncated
    if len(all_messages) > max_messages:
        messages_table.footer = (
            f"[dim]Showing last {max_messages} of {len(all_messages)} messages[/dim]"
        )

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                safe_markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    tool_calls_count = len(message_buffer.tool_calls)
    llm_calls_count = sum(
        1 for _, msg_type, _ in message_buffer.messages if msg_type == "Reasoning"
    )
    reports_count = sum(
        1 for content in message_buffer.report_sections.values() if content is not None
    )

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(
        f"Tool Calls: {tool_calls_count} | LLM Calls: {llm_calls_count} | Generated Reports: {reports_count}"
    )

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


def get_user_selections():
    """Get all user selections before starting the analysis display."""
    # Display ASCII art welcome message
    with open("./cli/static/welcome.txt", "r") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]Deep Value Intelligence: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to Deep Value Intelligence",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()  # Add a blank line after the welcome box

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # Step 1: Ticker symbol
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
        )
    )
    selected_ticker = get_ticker()

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Select analysts
    console.print(
        create_question_box(
            "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth
    console.print(
        create_question_box(
            "Step 4: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 5: OpenAI backend
    console.print(
        create_question_box(
            "Step 5: OpenAI backend", "Select which service to talk to"
        )
    )
    selected_llm_provider, backend_url = select_llm_provider()
    
    # Step 6: Thinking agents
    console.print(
        create_question_box(
            "Step 6: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    return {
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
    }


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    # Use a default date that's more likely to work with LLM APIs
    # Use December 2024 as a safe default that most APIs should support
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    while True:
        date_str = typer.prompt(
            "Enter analysis date (YYYY-MM-DD)", 
            default=default_date
        )
        try:
            # Validate date format
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            
            # Check if date is in the future
            if analysis_date > datetime.datetime.now():
                console.print("[yellow]Warning: Future dates may not have available data. Analysis might fail.[/yellow]")
                continue_anyway = typer.confirm("Continue with this future date anyway?")
                if not continue_anyway:
                    continue
            
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def _filter_trading_recommendations(content: str) -> str:
    """
    Filter out explicit BUY/SELL/HOLD recommendations from content for the AI Executive Summary.
    Removes lines containing explicit trading decisions while preserving the analysis.
    """
    import re

    if not content:
        return content

    lines = content.split('\n')
    filtered_lines = []

    # Patterns to filter out (case-insensitive)
    skip_patterns = [
        r'^\s*\*{0,2}(FINAL\s+)?(DECISION|RECOMMENDATION|VERDICT)\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD|STRONG\s+BUY|STRONG\s+SELL)',
        r'^\s*\*{0,2}(The\s+)?(recommendation|decision)\s+(is\s+to\s+)?(BUY|SELL|HOLD)',
        r'^\s*\*{0,2}(TRANSACTION\s+PROPOSAL)\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)',
        r'^\s*-?\s*\*{0,2}Action\s*:\s*\*{0,2}\s*(BUY|SELL|HOLD)',
        r'^\s*>\s*\*{0,2}(DECISION|RECOMMENDATION)\s*:\s*(BUY|SELL|HOLD)',
    ]

    for line in lines:
        skip_line = False
        for pattern in skip_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                skip_line = True
                break
        if not skip_line:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def generate_comprehensive_report(final_state, ticker, analysis_date, report_dir):
    """Generate a comprehensive markdown report combining all analysis."""
    from datetime import datetime

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
    report_lines.append("1. [AI Executive Summary](#ai-executive-summary)")
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

    # AI Executive Summary (with BUY/SELL/HOLD recommendations filtered out)
    report_lines.append("## AI Executive Summary")
    report_lines.append("")
    if final_state.get("final_trade_decision"):
        decision_content = _ensure_string(final_state["final_trade_decision"])
        # Filter out explicit BUY/SELL/HOLD recommendations
        filtered_content = _filter_trading_recommendations(decision_content)
        report_lines.append(filtered_content)
    else:
        report_lines.append("*No executive summary available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Analyst Team Reports Section
    report_lines.append("## Analyst Team Reports")
    report_lines.append("")

    # Fundamentals Analysis
    report_lines.append("### Fundamentals Analysis")
    report_lines.append("")
    if final_state.get("fundamentals_report"):
        report_lines.append(_ensure_string(final_state["fundamentals_report"]))
    else:
        report_lines.append("*No fundamentals analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Value Analysis (Buffett)
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

    # Growth Analysis (Lynch/Druckenmiller/Fisher)
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

    # Market Analysis
    report_lines.append("### Market Analysis")
    report_lines.append("")
    if final_state.get("market_report"):
        report_lines.append(_ensure_string(final_state["market_report"]))
    else:
        report_lines.append("*No market analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # Social Sentiment Analysis
    report_lines.append("### Social Sentiment Analysis")
    report_lines.append("")
    if final_state.get("sentiment_report"):
        report_lines.append(_ensure_string(final_state["sentiment_report"]))
    else:
        report_lines.append("*No sentiment analysis available.*")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # News Analysis
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

    # Write the comprehensive report
    comprehensive_report = "\n".join(report_lines)
    report_path = report_dir / f"{ticker}_deep_value_intelligence_{analysis_date}.md"
    with open(report_path, "w") as f:
        f.write(comprehensive_report)

    return report_path


def display_complete_report(final_state):
    """Display the complete analysis report with team-based panels."""
    console.print("\n[bold green]Complete Analysis Report[/bold green]\n")

    # I. Analyst Team Reports
    analyst_reports = []

    # Fundamentals Analyst Report
    if final_state.get("fundamentals_report"):
        analyst_reports.append(
            Panel(
                safe_markdown(final_state["fundamentals_report"]),
                title="Fundamentals Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Value Analyst Report (Buffett methodology)
    if final_state.get("value_report"):
        analyst_reports.append(
            Panel(
                safe_markdown(final_state["value_report"]),
                title="Value Analyst (Buffett)",
                border_style="magenta",
                padding=(1, 2),
            )
        )

    # Growth Analyst Report (Lynch/Druckenmiller/Fisher methodology)
    if final_state.get("growth_report"):
        analyst_reports.append(
            Panel(
                safe_markdown(final_state["growth_report"]),
                title="Growth Analyst (Lynch/Druckenmiller/Fisher)",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Market Analyst Report
    if final_state.get("market_report"):
        analyst_reports.append(
            Panel(
                safe_markdown(final_state["market_report"]),
                title="Market Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # Social Analyst Report
    if final_state.get("sentiment_report"):
        analyst_reports.append(
            Panel(
                safe_markdown(final_state["sentiment_report"]),
                title="Social Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    # News Analyst Report
    if final_state.get("news_report"):
        analyst_reports.append(
            Panel(
                safe_markdown(final_state["news_report"]),
                title="News Analyst",
                border_style="blue",
                padding=(1, 2),
            )
        )

    if analyst_reports:
        console.print(
            Panel(
                Columns(analyst_reports, equal=True, expand=True),
                title="I. Analyst Team Reports",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        research_reports = []
        debate_state = final_state["investment_debate_state"]

        # Bull Researcher Analysis
        if debate_state.get("bull_history"):
            research_reports.append(
                Panel(
                    safe_markdown(debate_state["bull_history"]),
                    title="Bull Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Bear Researcher Analysis
        if debate_state.get("bear_history"):
            research_reports.append(
                Panel(
                    safe_markdown(debate_state["bear_history"]),
                    title="Bear Researcher",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Research Manager Decision
        if debate_state.get("judge_decision"):
            research_reports.append(
                Panel(
                    safe_markdown(debate_state["judge_decision"]),
                    title="Research Manager",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if research_reports:
            console.print(
                Panel(
                    Columns(research_reports, equal=True, expand=True),
                    title="II. Research Team Decision",
                    border_style="magenta",
                    padding=(1, 2),
                )
            )

    # III. Trading Team Reports
    if final_state.get("trader_investment_plan"):
        console.print(
            Panel(
                Panel(
                    safe_markdown(final_state["trader_investment_plan"]),
                    title="Trader",
                    border_style="blue",
                    padding=(1, 2),
                ),
                title="III. Trading Team Plan",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # IV. Risk Management Team Reports
    if final_state.get("risk_debate_state"):
        risk_reports = []
        risk_state = final_state["risk_debate_state"]

        # Aggressive (Risky) Analyst Analysis
        if risk_state.get("risky_history"):
            risk_reports.append(
                Panel(
                    safe_markdown(risk_state["risky_history"]),
                    title="Aggressive Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Conservative (Safe) Analyst Analysis
        if risk_state.get("safe_history"):
            risk_reports.append(
                Panel(
                    safe_markdown(risk_state["safe_history"]),
                    title="Conservative Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Neutral Analyst Analysis
        if risk_state.get("neutral_history"):
            risk_reports.append(
                Panel(
                    safe_markdown(risk_state["neutral_history"]),
                    title="Neutral Analyst",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        if risk_reports:
            console.print(
                Panel(
                    Columns(risk_reports, equal=True, expand=True),
                    title="IV. Risk Management Team",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # V. Deep Values Strategy
        if risk_state.get("judge_decision"):
            console.print(
                Panel(
                    Panel(
                        safe_markdown(risk_state["judge_decision"]),
                        title="Portfolio Manager",
                        border_style="blue",
                        padding=(1, 2),
                    ),
                    title="V. Deep Values Strategy",
                    border_style="green",
                    padding=(1, 2),
                )
            )


def update_research_team_status(status):
    """Update status for all research team members and trader."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager", "Trader"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)

def extract_content_string(content):
    """Extract string content from various message formats, extracting only 'text' field."""
    if content is None:
        return ""
    if isinstance(content, str):
        # Check if it's a string representation of a dict/list
        stripped = content.strip()
        if stripped.startswith('{') or stripped.startswith('['):
            try:
                import ast
                parsed = ast.literal_eval(content)
                return extract_content_string(parsed)
            except (ValueError, SyntaxError):
                pass
        return content
    elif isinstance(content, dict):
        # Handle single dict: {'type': 'text', 'text': '...', 'extras': {...}}
        if content.get('type') == 'text' and 'text' in content:
            return content['text']
        elif content.get('type') == 'tool_use':
            return f"[Tool: {content.get('name', 'unknown')}]"
        return str(content)
    elif isinstance(content, list):
        # Handle Gemini/Anthropic's list format
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif item.get('type') == 'tool_use':
                    text_parts.append(f"[Tool: {item.get('name', 'unknown')}]")
            else:
                text_parts.append(str(item))
        return '\n\n'.join(text_parts)
    else:
        return str(content)

def safe_markdown(content):
    """Safely create a Markdown object, ensuring content is a string."""
    if content is None:
        return Markdown("")
    return Markdown(extract_content_string(content))

def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()

    # Initialize the graph
    graph = TradingAgentsGraph(
        [analyst.value for analyst in selections["analysts"]], config=config, debug=True
    )

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper
    
    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        # Map section names to file names (rename final_trade_decision to AI_investing_strategy)
        file_name_mapping = {
            "final_trade_decision": "AI_investing_strategy",
        }
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    base_name = file_name_mapping.get(section_name, section_name)
                    file_name = f"{base_name}.md"
                    with open(report_dir / file_name, "w") as f:
                        # Use _ensure_string to extract proper text from Gemini's list format
                        content_str = _ensure_string(content)
                        f.write(content_str)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout)

        # Reset agent statuses
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "pending")

        # Reset report sections
        for section in message_buffer.report_sections:
            message_buffer.report_sections[section] = None
        message_buffer.current_report = None
        message_buffer.final_report = None

        # Update agent status to in_progress for the first analyst
        first_analyst = f"{selections['analysts'][0].value.capitalize()} Analyst"
        message_buffer.update_agent_status(first_analyst, "in_progress")
        update_display(layout)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text)

        # Initialize state and get graph args
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args()

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                # Get the last message from the chunk
                last_message = chunk["messages"][-1]

                # Extract message content and type
                if hasattr(last_message, "content"):
                    content = extract_content_string(last_message.content)  # Use the helper function
                    msg_type = "Reasoning"
                else:
                    content = str(last_message)
                    msg_type = "System"

                # Add message to buffer
                message_buffer.add_message(msg_type, content)                

                # If it's a tool call, add it to tool calls
                if hasattr(last_message, "tool_calls"):
                    for tool_call in last_message.tool_calls:
                        # Handle both dictionary and object tool calls
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(
                                tool_call["name"], tool_call["args"]
                            )
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

                # Update reports and agent status based on chunk content
                # Analyst Team Reports (order: fundamentals → value → growth → market → social → news)
                if "fundamentals_report" in chunk and chunk["fundamentals_report"]:
                    message_buffer.update_report_section(
                        "fundamentals_report", chunk["fundamentals_report"]
                    )
                    message_buffer.update_agent_status(
                        "Fundamentals Analyst", "completed"
                    )
                    # Set next analyst to in_progress
                    if "value" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Value Analyst", "in_progress"
                        )
                    elif "growth" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Growth Analyst", "in_progress"
                        )
                    elif "market" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Market Analyst", "in_progress"
                        )

                if "value_report" in chunk and chunk["value_report"]:
                    message_buffer.update_report_section(
                        "value_report", chunk["value_report"]
                    )
                    message_buffer.update_agent_status("Value Analyst", "completed")
                    # Set next analyst to in_progress
                    if "growth" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Growth Analyst", "in_progress"
                        )
                    elif "market" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Market Analyst", "in_progress"
                        )

                if "growth_report" in chunk and chunk["growth_report"]:
                    message_buffer.update_report_section(
                        "growth_report", chunk["growth_report"]
                    )
                    message_buffer.update_agent_status("Growth Analyst", "completed")
                    # Set next analyst to in_progress
                    if "market" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Market Analyst", "in_progress"
                        )

                if "market_report" in chunk and chunk["market_report"]:
                    message_buffer.update_report_section(
                        "market_report", chunk["market_report"]
                    )
                    message_buffer.update_agent_status("Market Analyst", "completed")
                    # Set next analyst to in_progress
                    if "social" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "Social Analyst", "in_progress"
                        )

                if "sentiment_report" in chunk and chunk["sentiment_report"]:
                    message_buffer.update_report_section(
                        "sentiment_report", chunk["sentiment_report"]
                    )
                    message_buffer.update_agent_status("Social Analyst", "completed")
                    # Set next analyst to in_progress
                    if "news" in selections["analysts"]:
                        message_buffer.update_agent_status(
                            "News Analyst", "in_progress"
                        )
                    else:
                        # No more analysts, move to research team
                        update_research_team_status("in_progress")

                if "news_report" in chunk and chunk["news_report"]:
                    message_buffer.update_report_section(
                        "news_report", chunk["news_report"]
                    )
                    message_buffer.update_agent_status("News Analyst", "completed")
                    # All analysts done, move to research team
                    update_research_team_status("in_progress")

                # Research Team - Handle Investment Debate State
                if (
                    "investment_debate_state" in chunk
                    and chunk["investment_debate_state"]
                ):
                    debate_state = chunk["investment_debate_state"]

                    # Update Bull Researcher status and report
                    if "bull_history" in debate_state and debate_state["bull_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bull response
                        bull_responses = debate_state["bull_history"].split("\n")
                        latest_bull = bull_responses[-1] if bull_responses else ""
                        if latest_bull:
                            message_buffer.add_message("Reasoning", latest_bull)
                            # Update research report with bull's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"### Bull Researcher Analysis\n{latest_bull}",
                            )

                    # Update Bear Researcher status and report
                    if "bear_history" in debate_state and debate_state["bear_history"]:
                        # Keep all research team members in progress
                        update_research_team_status("in_progress")
                        # Extract latest bear response
                        bear_responses = debate_state["bear_history"].split("\n")
                        latest_bear = bear_responses[-1] if bear_responses else ""
                        if latest_bear:
                            message_buffer.add_message("Reasoning", latest_bear)
                            # Update research report with bear's latest analysis
                            message_buffer.update_report_section(
                                "investment_plan",
                                f"{message_buffer.report_sections['investment_plan']}\n\n### Bear Researcher Analysis\n{latest_bear}",
                            )

                    # Update Research Manager status and final decision
                    if (
                        "judge_decision" in debate_state
                        and debate_state["judge_decision"]
                    ):
                        # Keep all research team members in progress until final decision
                        update_research_team_status("in_progress")
                        message_buffer.add_message(
                            "Reasoning",
                            f"Research Manager: {debate_state['judge_decision']}",
                        )
                        # Update research report with final decision
                        message_buffer.update_report_section(
                            "investment_plan",
                            f"{message_buffer.report_sections['investment_plan']}\n\n### Research Manager Decision\n{debate_state['judge_decision']}",
                        )
                        # Mark all research team members as completed
                        update_research_team_status("completed")
                        # Set first risk analyst to in_progress
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )

                # Trading Team
                if (
                    "trader_investment_plan" in chunk
                    and chunk["trader_investment_plan"]
                ):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    # Set first risk analyst to in_progress
                    message_buffer.update_agent_status("Risky Analyst", "in_progress")

                # Risk Management Team - Handle Risk Debate State
                if "risk_debate_state" in chunk and chunk["risk_debate_state"]:
                    risk_state = chunk["risk_debate_state"]

                    # Update Risky Analyst status and report
                    if (
                        "current_risky_response" in risk_state
                        and risk_state["current_risky_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Risky Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Risky Analyst: {risk_state['current_risky_response']}",
                        )
                        # Update risk report with risky analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Risky Analyst Analysis\n{risk_state['current_risky_response']}",
                        )

                    # Update Safe Analyst status and report
                    if (
                        "current_safe_response" in risk_state
                        and risk_state["current_safe_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Safe Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Safe Analyst: {risk_state['current_safe_response']}",
                        )
                        # Update risk report with safe analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Safe Analyst Analysis\n{risk_state['current_safe_response']}",
                        )

                    # Update Neutral Analyst status and report
                    if (
                        "current_neutral_response" in risk_state
                        and risk_state["current_neutral_response"]
                    ):
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Neutral Analyst: {risk_state['current_neutral_response']}",
                        )
                        # Update risk report with neutral analyst's latest analysis only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Neutral Analyst Analysis\n{risk_state['current_neutral_response']}",
                        )

                    # Update Portfolio Manager status and final strategy
                    if "judge_decision" in risk_state and risk_state["judge_decision"]:
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "in_progress"
                        )
                        message_buffer.add_message(
                            "Reasoning",
                            f"Portfolio Manager: {risk_state['judge_decision']}",
                        )
                        # Update risk report with final strategy only
                        message_buffer.update_report_section(
                            "final_trade_decision",
                            f"### Portfolio Manager Strategy\n{risk_state['judge_decision']}",
                        )
                        # Mark risk analysts as completed
                        message_buffer.update_agent_status("Risky Analyst", "completed")
                        message_buffer.update_agent_status("Safe Analyst", "completed")
                        message_buffer.update_agent_status(
                            "Neutral Analyst", "completed"
                        )
                        message_buffer.update_agent_status(
                            "Portfolio Manager", "completed"
                        )

                # Update the display
                update_display(layout)

            trace.append(chunk)

        # Get final state
        final_state = trace[-1]

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "Analysis", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        # Generate comprehensive report
        comprehensive_report_path = generate_comprehensive_report(
            final_state,
            selections["ticker"],
            selections["analysis_date"],
            report_dir
        )
        console.print(f"\n[green]Comprehensive report saved to:[/green] {comprehensive_report_path}")

        # Display the complete final report
        display_complete_report(final_state)

        update_display(layout)


@app.command()
def analyze():
    run_analysis()


if __name__ == "__main__":
    app()
