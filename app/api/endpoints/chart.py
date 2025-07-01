from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import io
import matplotlib.pyplot as plt
import yfinance as yf
from app.services.utils import map_ticker
import sqlite3

router = APIRouter()

@router.get("/chart")
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
