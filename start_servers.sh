#!/bin/bash

# Start both FastAPI and Streamlit servers
# FastAPI runs on port 8000
# Streamlit runs on port 8501

echo "Starting Deep-Research Backend Services..."
echo "=========================================="

# Activate virtual environment if it exists
if [ -f "/app/.venv/bin/activate" ]; then
    source /app/.venv/bin/activate
fi

# Start FastAPI server in background
echo "Starting FastAPI server on http://0.0.0.0:8000"
ddtrace-run uvicorn packages.agents.src.agent_api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
echo "FastAPI PID: $FASTAPI_PID"

# Wait a moment for FastAPI to start
sleep 2

# Start Streamlit server in background
echo "Starting Streamlit server on http://0.0.0.0:8501"
ddtrace-run streamlit run apps/ui/src/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true &
STREAMLIT_PID=$!
echo "Streamlit PID: $STREAMLIT_PID"

echo "=========================================="
echo "Services started successfully!"
echo "=========================================="

# Keep the script running and handle graceful shutdown
trap "kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true" EXIT

wait
