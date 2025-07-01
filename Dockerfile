# Use a Python base that matches what you want (3.13 still, if you need it)
FROM python:3.13-slim

# Install system-level build deps for numpy/pandas
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gfortran \
      libatlas-base-dev \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Expose port and launch Uvicorn
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
