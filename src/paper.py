"""Paper-trading: cuenta simulada $10,000. Solo LONG/SIN POSICION.
Comision 0.075% por lado (Binance spot aprox).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperAccount:
    cash: float = 10000.0
    qty: float = 0.0
    fee: float = 0.00075
    trades: list = field(default_factory=list)

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def buy(self, price: float, ts: str = "") -> dict:
        if self.qty > 0 or self.cash <= 0:
            return {"ok": False, "why": "ya en posicion o sin cash"}
        q = (self.cash / price) * (1 - self.fee)
        cost = q * price * (1 + self.fee)
        cost = min(cost, self.cash)
        q = (cost / price) * (1 - self.fee)
        self.qty, self.cash = q, self.cash - cost
        t = {"ts": ts, "side": "BUY", "price": price, "qty": q}
        self.trades.append(t)
        return {"ok": True, **t}

    def sell(self, price: float, ts: str = "") -> dict:
        if self.qty <= 0:
            return {"ok": False, "why": "sin posicion"}
        proceeds = self.qty * price * (1 - self.fee)
        t = {"ts": ts, "side": "SELL", "price": price, "qty": self.qty, "proceeds": proceeds}
        self.cash, self.qty = self.cash + proceeds, 0.0
        self.trades.append(t)
        return {"ok": True, **t}
