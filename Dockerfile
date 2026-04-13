FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements_api.txt -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY data/processed/ ./data/processed/

# models/ and api/swing_history.db are provided via volume mounts at runtime

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
