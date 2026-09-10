# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .

# Install system libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Expose API port
EXPOSE 8000

# Start API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
