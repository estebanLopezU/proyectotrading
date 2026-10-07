"""Paper-trading: cuenta simulada $10,000. Solo LONG/SIN POSICION.
Comision 0.075% por lado (Binance spot aprox).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PaperAccount:
    cash: float = 10000.0
    qty: float = 0.0
    fee: float = 0.00075
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)  # equity history

    def equity(self, price: float) -> float:
        return self.cash + self.qty * price

    def record_equity(self, price: float, ts: str = "") -> None:
        """Registra equity actual en la curva."""
        self.equity_curve.append({
            "ts": ts or datetime.now(timezone.utc).isoformat(),
            "price": price,
            "equity": self.equity(price),
            "cash": self.cash,
            "qty": self.qty,
        })

    def buy(self, price: float, ts: str = "", strategy: str = "rule", rr: float = 0.0) -> dict:
        if self.qty > 0 or self.cash <= 0:
            return {"ok": False, "why": "ya en posicion o sin cash"}
        # Calculate quantity: cash * (1 - fee) / price, then cost = qty * price * (1 + fee)
        # The min ensures cost never exceeds available cash (safety cap)
        q = (self.cash / price) * (1 - self.fee)
        cost = q * price * (1 + self.fee)
        cost = min(cost, self.cash)
        q = (cost / price) * (1 - self.fee)
        self.qty, self.cash = q, self.cash - cost
        t = {
            "ts": ts or datetime.now(timezone.utc).isoformat(),
            "side": "BUY",
            "price": price,
            "qty": q,
            "strategy": strategy,
            "rr": rr,
            "status": "OPEN",
            "pnl": 0.0,
            "pnl_pct": 0.0,
        }
        self.trades.append(t)
        return {"ok": True, **t}

    def sell(self, price: float, ts: str = "", strategy: str = "", exit_reason: str = "") -> dict:
        if self.qty <= 0:
            return {"ok": False, "why": "sin posicion"}
        proceeds = self.qty * price * (1 - self.fee)
        entry_price = 0.0
        entry_ts = ""
        entry_strategy = strategy
        entry_rr = 0.0
        for t in reversed(self.trades):
            if t.get("side") == "BUY" and t.get("status") == "OPEN":
                entry_price = t["price"]
                entry_ts = t["ts"]
                entry_strategy = t.get("strategy", strategy)
                entry_rr = t.get("rr", 0.0)
                t["status"] = "CLOSED"
                break
        pnl = proceeds - (self.qty * entry_price * (1 + self.fee))
        pnl_pct = (price / entry_price - 1) * 100 if entry_price > 0 else 0.0
        t = {
            "ts": ts or datetime.now(timezone.utc).isoformat(),
            "side": "SELL",
            "price": price,
            "qty": self.qty,
            "proceeds": proceeds,
            "entry_price": entry_price,
            "entry_ts": entry_ts,
            "strategy": entry_strategy,
            "rr": entry_rr,
            "status": "CLOSED",
            "exit_reason": exit_reason or "signal",
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }
        self.cash, self.qty = self.cash + proceeds, 0.0
        self.trades.append(t)
        return {"ok": True, **t}

    def get_trades_df(self):
        """Devuelve DataFrame de trades cerrados."""
        import pandas as pd
        closed = [t for t in self.trades if t.get("status") == "CLOSED" and t.get("side") == "SELL"]
        if not closed:
            return pd.DataFrame()
        return pd.DataFrame(closed)

    def stats(self) -> dict:
        """Calcula estadisticas basicas de trades cerrados."""
        import pandas as pd
        df = self.get_trades_df()
        if df.empty:
            return {"n": 0, "win_rate": 0, "profit_factor": 0, "total_pnl": 0.0, "avg_rr": 0}
        wins = df[df["pnl"] > 0]
        losses = df[df["pnl"] <= 0]
        n = len(df)
        win_rate = len(wins) / n if n > 0 else 0
        gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["pnl"].sum()) if not losses.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        return {
            "n": n,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl": df["pnl"].sum(),
            "avg_rr": df["rr"].mean() if "rr" in df.columns else 0,
            "wins": len(wins),
            "losses": len(losses),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }
