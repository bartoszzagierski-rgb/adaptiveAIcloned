"""Integracja TradingGym z MetaTrader 5, trening RL, LLM (Ollama) i handel demo/live."""
import os

# pandas/tables uzywa NumExpr; bez tego na 20-watkowym CPU wypisuje ostrzezenie i ogranicza sie do 16
os.environ.setdefault('NUMEXPR_MAX_THREADS', str(os.cpu_count() or 1))
