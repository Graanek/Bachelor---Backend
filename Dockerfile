FROM python:3.11-slim

# (optional: if you still need any native deps later)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
   
EXPOSE 8000
# assume your FastAPI app is in app/main.py and exposes `app = FastAPI()`
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]