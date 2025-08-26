# Stage 1 — Build
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential cmake libpq-dev libboost-all-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libopenblas-dev liblapack-dev \
    gfortran curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2 — Runtime only
FROM python:3.11-slim
WORKDIR /app

# Install runtime dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg libpq-dev \
    libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
    libopenblas-dev liblapack-dev  \
    iproute2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Create user FIRST (before creating directories)
RUN useradd --create-home --shell /bin/bash appuser

# Create directories with proper ownership
RUN mkdir -p /app/videos /app/temp_videos /app/logs /app/data && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER appuser

# Copy application files with proper ownership
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PYTHONPATH="/app"

# Expose port for websocket
EXPOSE 8765

# Use exec form for better signal handling
ENTRYPOINT ["python", "main.py"]