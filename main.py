# main.py
import io
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tradingview_ta import TA_Handler, Interval

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Konfiguracja CORS – pozwalamy na połączenia z dowolnego źródła
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Domyślna lista 10 najpopularniejszych kryptowalut (bez stablecoinów)
DEFAULT_ASSETS = ["BTC", "ETH", "XRP", "SOL"]

def map_ticker(asset: str) -> str:
    """Mapuje symbol kryptowaluty na ticker Yahoo Finance (np. BTC -> BTC-USD)."""
    return f"{asset}-USD"

@app.get("/portfolio")
async def portfolio(
    start_date: str = Query(..., description="Data początkowa w formacie RRRR-MM-DD"),
    end_date: str = Query(..., description="Data końcowa w formacie RRRR-MM-DD"),
    assets: str = Query(
        None,
        description="Lista aktywów oddzielona przecinkami (np. BTC,ETH,...). Jeśli puste, użyjemy domyślnej listy 10 kryptowalut."
    )
):
    # Jeśli nie podano aktywów, używamy domyślnej listy
    if assets is None or assets.strip() == "":
        asset_list = DEFAULT_ASSETS
    else:
        asset_list = [asset.strip().upper() for asset in assets.split(",") if asset.strip() != ""]

    # Parsowanie dat
    try:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Błędny format daty. Użyj RRRR-MM-DD.")

    # Mapowanie aktywów na tickery
    tickers = [map_ticker(asset) for asset in asset_list]

    # Pobieranie danych z yfinance
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by="column")
        if "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]
        else:
            raise HTTPException(status_code=500, detail="Brak kolumny 'Adj Close' lub 'Close' w danych.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd pobierania danych: {e}")

    data = data.dropna()
    if data.empty:
        raise HTTPException(status_code=404, detail="Brak danych dla podanego zakresu dat.")

    # Obliczanie dziennych zwrotów
    returns = data.pct_change().dropna()
    if returns.empty:
        raise HTTPException(status_code=404, detail="Brak wystarczających danych do obliczeń zwrotów.")

    # Ustalanie liczby symulacji (minimum 500)
    days = (end - start).days
    num_portfolios = 1000
    if num_portfolios < 500:
        num_portfolios = 500

    n = num_portfolios
    num_assets = len(tickers)
    random_weights = np.random.random((n, num_assets))
    random_weights = random_weights / random_weights.sum(axis=1, keepdims=True)

    # Obliczenia zwrotów portfela i zmienności
    mean_daily_returns = returns.mean()
    port_returns = (random_weights @ mean_daily_returns.values) * 252
    port_vols = np.sqrt(np.sum(random_weights * (random_weights @ returns.cov()), axis=1) * 252)
    sharpes = np.divide(port_returns, port_vols, out=np.zeros_like(port_returns), where=port_vols != 0)

    best_idx = np.argmax(sharpes)
    best_weights = random_weights[best_idx]
    best_allocation = {asset: round(weight * 100, 2) for asset, weight in zip(asset_list, best_weights)}

    # Obliczanie efficient frontier
    sorted_indices = np.argsort(port_vols)
    sorted_vols = port_vols[sorted_indices]
    sorted_rets = port_returns[sorted_indices]
    efficient_vols = []
    efficient_rets = []
    current_max_ret = -np.inf
    for vol, ret in zip(sorted_vols, sorted_rets):
        if ret > current_max_ret:
            efficient_vols.append(vol)
            efficient_rets.append(ret)
            current_max_ret = ret

    # Próbkowanie symulacji (aby nie wysyłać zbyt wielu punktów)
    scatter_sample_size = 2000
    if n > scatter_sample_size:
        indices = np.random.choice(n, scatter_sample_size, replace=False)
        sim_vols_sample = port_vols[indices]
        sim_rets_sample = port_returns[indices]
        sim_sharpes_sample = sharpes[indices]
    else:
        sim_vols_sample = port_vols
        sim_rets_sample = port_returns
        sim_sharpes_sample = sharpes

    response_data = {
        "simulations": {
            "vols": sim_vols_sample.tolist(),
            "rets": sim_rets_sample.tolist(),
            "sharpes": sim_sharpes_sample.tolist()
        },
        "efficient_frontier": {
            "vols": efficient_vols,
            "rets": efficient_rets
        },
        "best_allocation": best_allocation,
        "num_simulations": num_portfolios,
        "assets": asset_list
    }

    return JSONResponse(content=response_data)

@app.get("/price-data")
async def price_data(
    asset: str = Query(..., description="Symbol aktywa, np. BTC, ETH"),
    timeframe: str = Query(..., description="Przedział czasowy: 1d, 1w, 1m, all")
):
    try:
        # Map timeframe to TradingView interval
        interval_map = {
            "1d": Interval.INTERVAL_1_MINUTE,
            "1w": Interval.INTERVAL_1_HOUR,
            "1m": Interval.INTERVAL_4_HOURS,
            "all": Interval.INTERVAL_1_DAY
        }
        
        interval = interval_map.get(timeframe, Interval.INTERVAL_1_HOUR)
        
        # Initialize TradingView handler
        handler = TA_Handler(
            symbol=f"{asset}USDT",
            screener="crypto",
            exchange="BINANCE",
            interval=interval
        )
        
        # Get analysis from TradingView
        analysis = handler.get_analysis()
        
        # Get current price and indicators
        current_price = analysis.indicators["close"]
        
        # For 24h change calculation
        open_price = analysis.indicators["open"]
        high = analysis.indicators["high"]
        low = analysis.indicators["low"]
        
        # Get RSI and MACD values
        rsi = analysis.indicators['RSI']
        macd = {
            'MACD': analysis.indicators['MACD.macd'],
            'Signal': analysis.indicators['MACD.signal'],
            'Histogram': analysis.indicators['MACD.hist']
        }
        
        
        # Calculate 24h change
        price_change = ((current_price - open_price) / open_price) * 100
        
        # Create response data
        response_data = {
            "dates": [datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
            "prices": [current_price],
            "price_change_24h": price_change,
            "high_24h": high,
            "low_24h": low,
            "technical_indicators": {
                "rsi": rsi,
                "macd": macd
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")

# Nowy endpoint generujący wykres jako obraz (matplotlib)
@app.get("/chart")
async def chart(
    symbol: str = Query("BTC", description="Symbol aktywa, np. BTC, ETH"),
    timeframe: str = Query("1d", description="Przedział czasowy: 1d, 1w, 1m, all")
):
    ticker = map_ticker(symbol)
    
    # Ustalanie okresu i interwału w zależności od timeframe
    if timeframe == "1d":
        period = "1d"
        interval = "1m"
    elif timeframe == "1w":
        period = "5d"   # pobieramy dane z ostatnich 5 dni
        interval = "5m"
    elif timeframe == "1m":
        period = "1mo"
        interval = "30m"
    elif timeframe == "1y":
        period = "1y"
        interval = "1d"
    elif timeframe == "all":
        period = "max"
        interval = "1d"
    else:
        raise HTTPException(status_code=400, detail="Niepoprawny timeframe. Użyj: 1d, 1w, 1m lub all.")
    
    # Pobieranie danych
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise HTTPException(status_code=404, detail="Brak danych dla wybranego aktywa i przedziału czasowego.")
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise HTTPException(status_code=500, detail="Brak odpowiednich kolumn w danych.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(prices.index, prices, color="red", linewidth=2)
    plt.title(f"{symbol.upper()} Price Chart ({timeframe})")
    plt.xlabel("Data")
    plt.ylabel("Cena (USD)")
    plt.grid(True)
    
    if timeframe == "all":
        plt.yscale("log")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
