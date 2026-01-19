"""
TradingAgents – default_config.py
Configured for a local Ollama backend running Llama-3 8B.

Prerequisites
-------------
$ ollama pull llama3:8b          # or llama3:8b-instruct, etc.
$ ollama pull nomic-embed-text   # optional, avoids first-run 404s
$ ollama serve                   # serves http://localhost:11434/v1

Recommended shell variables (add to ~/.zshrc, ~/.bashrc, etc.)
--------------------------------------------------------------
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"               # any non-empty string
# Optional per-session overrides:
# export TRADINGAGENTS_DEEP_MODEL="llama3:8b"
# export TRADINGAGENTS_QUICK_MODEL="llama3:8b"
"""

import os
from pathlib import Path

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    # API configuration
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "openai_api_base": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    "google_api_base": os.getenv("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
}

#ROOT_DIR = Path(__file__).resolve().parent
#
#DEFAULT_CONFIG = {
#    # ─── Paths ───────────────────────────────────────────────────────────
#    "project_dir": str(ROOT_DIR),
#    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
#    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
#    "data_cache_dir": str(ROOT_DIR / "dataflows" / "data_cache"),
#
#    # ─── LLM / Embedding backend ─────────────────────────────────────────
#    "llm_provider": "openai",                      # keep the OpenAI SDK façade
#    "backend_url": os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
#
#    # Primary reasoning models
#    "deep_think_llm":  os.getenv("TRADINGAGENTS_DEEP_MODEL",  "llama3.2"),
#    "quick_think_llm": os.getenv("TRADINGAGENTS_QUICK_MODEL", "llama3.2"),
#
#    # ─── Hyper-parameters ────────────────────────────────────────────────
#    "max_debate_rounds": 1,
#    "max_risk_discuss_rounds": 1,
#    "max_recur_limit": 100,
#
#    # ─── Tooling switches ────────────────────────────────────────────────
#    "online_tools": True,            # set False for fully offline runs
#}
