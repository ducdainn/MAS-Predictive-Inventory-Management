# BrickDemand UI - Dockerfile
# Multi-stage build for optimized image size

# ============================================
# Stage 1: Build stage - Install dependencies
# ============================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt requirements_ml.txt requirements_streamlit.txt ./

# Install all requirements
RUN pip install --no-cache-dir --user \
    -r requirements.txt \
    -r requirements_ml.txt \
    -r requirements_streamlit.txt \
    xgboost \
    plotly

# ============================================
# Stage 2: Runtime stage - Final image
# ============================================
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY agent/ ./agent/
COPY models_panel/ ./models_panel/
COPY init/ ./init/
COPY system_date_config.json ./

# Create directories for outputs and logs
RUN mkdir -p charts model_logs sql_logs workflow_logs memory

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Qdrant Cloud mode (requires QDRANT_URL and QDRANT_API_KEY)
ENV QDRANT_MODE=cloud

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "agent/ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=true"]


