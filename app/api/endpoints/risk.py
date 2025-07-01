from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from app.services.risk import compute_risk_metrics
import sqlite3

router = APIRouter()

@router.get("/risk-metrics")
async def risk_metrics(
    asset: str = Query(..., description="Symbol aktywa, np. BTC"),
    risk_free_rate: float = Query(0.0, description="Roczna stopa wolna od ryzyka")
):
    try:
        asset = asset.upper()
        metrics = compute_risk_metrics(asset, risk_free_rate)
        return JSONResponse(metrics)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating risk metrics: {e}")
