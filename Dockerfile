# Multi-stage build for production
FROM python:3.10-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements-railway.txt requirements.txt
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Railway uses PORT environment variable
ENV PORT=8000

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')"

# Run application with Railway's PORT
CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
