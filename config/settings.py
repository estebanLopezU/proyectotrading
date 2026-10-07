"""Fuente unica de verdad del prototipo. Nada hardcodeado en otros modulos."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1m")
    limit: int = int(os.getenv("LIMIT", "300"))
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")
    cache_path: str = os.getenv("CACHE_PATH", "data/btc_1m.csv")
    display_tz: str = os.getenv("DISPLAY_TZ", "America/Bogota")
    refresh_seconds: int = int(os.getenv("REFRESH_SECONDS", "15"))
    timeframes: tuple = ("1m", "5m", "15m", "1h")
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


SETTINGS = Settings()


def cache_path_for(symbol: str, timeframe: str) -> str:
    safe = symbol.replace("/", "_").replace(":", "_")
    return f"data/{safe}_{timeframe}.csv"
