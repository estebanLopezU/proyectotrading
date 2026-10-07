"""Prueba de alertas Telegram sin bloquear el dashboard.
Uso: python scripts/test_alert.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.alerts import send_telegram  # noqa: E402

if __name__ == "__main__":
    r = send_telegram("✅ Trading Prototype: prueba de alerta Telegram OK")
    print("RESULTADO:", r)
    if not r.get("ok"):
        print("Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env (ver .env.example)")