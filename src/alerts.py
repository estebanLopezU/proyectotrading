"""Alertas Telegram (opcional). Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID en .env.
Si no estan, solo imprime en log. Nunca bloquea el dashboard.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request


def send_telegram(msg: str) -> dict:
    tok = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat = os.getenv("TELEGRAM_CHAT_ID", "")
    if not tok or not chat:
        return {"ok": False, "why": "sin credenciales (ver .env.example)"}
    try:
        url = f"https://api.telegram.org/bot{tok}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return {"ok": r.status == 200, "status": r.status}
    except Exception as e:
        return {"ok": False, "why": str(e)[:150]}
