FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached as long as requirements.txt doesn't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model artifacts
COPY main.py generate_schema.py ./
COPY model/ model/

EXPOSE 8000

# Run with 2 workers; for CPU-bound XGBoost inference, more workers = more throughput
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
