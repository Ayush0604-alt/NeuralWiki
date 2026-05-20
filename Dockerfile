# Backend Dockerfile for Render
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY backend/app ./app

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Run uvicorn (environment variables come from Render dashboard)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
