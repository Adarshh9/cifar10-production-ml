# Multi-stage build for AWS deployment
FROM python:3.10-slim as builder

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create configs directory if needed
RUN mkdir -p configs

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
