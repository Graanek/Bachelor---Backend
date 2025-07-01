from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from datetime import date
import sqlite3
from app.services.sentiment import (
    get_cached_global,
    cache_global,
    get_cached_ta,
    cache_ta,
    fetch_fear_and_greed,
    fetch_vix,
    fetch_rsi_macd
)

router = APIRouter()

@router.get("/sentiment")
async def sentiment(asset: str = Query(..., description="e.g. BTC, ETH")):
    d = date.today().isoformat()

    # 1) Global
    global_data = get_cached_global(d)
    if not global_data:
        try:
            fg = fetch_fear_and_greed()
            print(f"Fetched Fear & Greed: {fg}")
            v = fetch_vix()
            print(f"Fetched VIX: {v}")
        except Exception as e:
            print(f"Global fetch error: {e}")
            raise HTTPException(502, detail=f"Global fetch error: {e}")
        cache_global(d, fg, v)
        global_data = {"fear_greed": fg, "vix": v}

    # 2) Per-asset
    asset = asset.upper()
    ta_data = get_cached_ta(d, asset)
    if not ta_data:
        try:
            ta_data = fetch_rsi_macd(asset)
            print(f"Fetched TA for {asset}: {ta_data}")
        except Exception as e:
            print(f"TA fetch error for {asset}: {e}")
            raise HTTPException(502, detail=f"TA fetch error for {asset}: {e}")
        cache_ta(d, asset, ta_data)

    # 3) Merge & return
    return JSONResponse({**global_data, **ta_data})
