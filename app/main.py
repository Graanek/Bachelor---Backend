from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import portfolio, price, sentiment, risk, chart
from app.database.models import init_db
from app.services.binance import download_all_data
from contextlib import asynccontextmanager 
import sqlite3
from app.config import DB

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logika uruchamiania aplikacji (startup)
    print("Initializing database...")
    conn = sqlite3.connect(DB) # Tworzenie połączenia z bazą danych
    init_db(conn) # Przekazanie połączenia do init_db
    conn.close() # Zamknięcie połączenia po inicjalizacji tabel

    print("Downloading all initial data...")
    download_all_data() # Ta funkcja już sama zarządza połączeniami

    print("Application startup complete.")
    yield # Aplikacja będzie działać, aż do jej zamknięcia
    # Logika zamykania aplikacji (shutdown) - jeśli potrzebna
    # Tutaj możesz zamknąć wszelkie połączenia lub zasoby, które były otwarte globalnie
    print("Application shutdown complete.")


app = FastAPI(lifespan=lifespan)
# Konfiguracja CORS – pozwalamy na połączenia z dowolnego źródła
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(portfolio.router, prefix="/api", tags=["Portfolio"])
app.include_router(price.router, prefix="/api", tags=["Price"])
app.include_router(sentiment.router, prefix="/api", tags=["Sentiment"])
app.include_router(risk.router, prefix="/api", tags=["Risk"])
app.include_router(chart.router, prefix="/api", tags=["Chart"])




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
