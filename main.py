from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Skonfiguruj CORS – dzięki temu frontend będzie mógł komunikować się z backendem
origins = [
    "http://localhost:3000",  # przykładowy adres React (jeśli korzystasz z Create React App)
    "http://localhost:5173",  # domyślny adres Vite
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # lista dozwolonych domen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World from FastAPI!"}
