import numpy as np
import pandas as pd
from datetime import datetime
from app.database.operations import get_price_history

def compute_risk_metrics(asset: str, risk_free_rate: float = 0.0) -> dict:
    """Calculate Sharpe, Sortino, Omega ratios, and Daily VaR (95%) for the given asset."""
    history = get_price_history(asset, "1d")
    if not history:
        raise ValueError(f"No data found for {asset}")

    # Zamieniamy historię na pandas Series
    dates = [datetime.strptime(entry[0], '%Y-%m-%d %H:%M:%S') for entry in history]
    prices = [entry[1] for entry in history]
    series = pd.Series(prices, index=dates).sort_index()
    
    # Obliczamy dzienne zwroty
    returns = series.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough data to calculate returns")

    # Ustalanie dziennej stopy wolnej od ryzyka
    rf_daily = risk_free_rate / 252
    excess = returns - rf_daily

    # 1) Sharpe Ratio (annualized)
    sharpe = 0.0
    if excess.std() != 0:
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252)

    # 2) Sortino Ratio (annualized)
    downside = returns[returns < rf_daily] - rf_daily
    sortino = 0.0
    if len(downside) > 0:
        downside_std = np.sqrt((downside ** 2).mean())
        if downside_std != 0:
            sortino = (excess.mean() / downside_std) * np.sqrt(252)

    # 3) Omega Ratio
    gains = returns[returns >= rf_daily] - rf_daily
    losses = rf_daily - returns[returns < rf_daily]
    omega = gains.sum() / losses.sum() if losses.sum() != 0 else float('inf')

    # 4) Daily VaR at 95% confidence
    #    Przyjmujemy VaR = - (5th percentile z dziennych zwrotów)
    #    Czyli VaR zawsze podajemy jako liczba dodatnia w procentach
    var_95 = 0.0
    try:
        percentile_5 = np.percentile(returns, 5)  # 5th percentile (wartość np. -0.03 oznacza stratę 3%)
        var_95 = -percentile_5 * 100  # przemnażamy przez 100, żeby uzyskać procenty
    except Exception:
        var_95 = 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "omega": omega,
        "price_change_pct": var_95
    }
