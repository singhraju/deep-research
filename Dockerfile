# =============================================================================
# Deep Research Healthcare Analytics Application - Production Dockerfile
# =============================================================================
# Single-stage build using UV package manager
# =============================================================================

FROM quay-nonprod.elevancehealth.com/programintegrity/python:3.12

WORKDIR /app

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"
ENV UV_NATIVE_TLS=1

# Copy corporate CA certificate bundle
# Store in /app for Python to use via SSL_CERT_FILE environment variable
COPY cacert.pem /app/cacert.pem

copy root_chain.pem ./

# Copy all application code
COPY apps/ ./apps/
COPY configs/ ./configs/
COPY packages/ ./packages/
COPY pyproject.toml ./
COPY version.py ./
COPY flexible_snowflake_connector-1.0.0-py3-none-any.whl ./

# Copy startup script with executable permissions
COPY --chmod=755 start_servers.sh /app/start_servers.sh

# Install dependencies directly in /app
RUN uv sync --no-dev

# Create directories for runtime data
RUN mkdir -p /app/logs /app/correlation_runs || true

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    SSL_CERT_FILE=/app/cacert.pem \
    REQUESTS_CA_BUNDLE=/app/cacert.pem

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Health check removed - base image may not support curl installation

# Run both FastAPI and Streamlit servers via start_servers.sh
ENTRYPOINT ["/app/start_servers.sh"]
