"""Utilidades: logs, formato moneda, fechas UTC -> display."""
import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from loguru import logger


def _safe_float(x, default: float = 0.0) -> float:
    """Convert to float, returning default if None or NaN."""
    try:
        fx = float(x)
    except (TypeError, ValueError):
        return default
    return fx if not math.isnan(fx) else default


def setup_logger():
    logger.remove()
    logger.add(
        sink=lambda m: print(m, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n",
        level="INFO",
    )
    return logger


def fmt_money(value: float, symbol: str = "USDT") -> str:
    try:
        return f"${value:,.2f} {symbol}"
    except Exception:
        return str(value)


def fmt_pct(value: float) -> str:
    try:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"
    except Exception:
        return str(value)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_display_tz(dt: datetime, tz_name: str = "America/Bogota"):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo(tz_name))
