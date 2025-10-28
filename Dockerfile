# Optimized Dockerfile for smaller image
FROM python:3.10-slim as builder

WORKDIR /app

# Install only necessary build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch CPU-only (much smaller)
RUN pip install --user --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install other dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy only installed packages
COPY --from=builder /root/.local /root/.local

# Copy application
COPY src/ ./src/
COPY models/ ./models/

RUN mkdir -p configs

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

# Reduce workers to 1 for t2.micro
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
