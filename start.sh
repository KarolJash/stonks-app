#!/bin/bash

# Exit immediately if any command returns a non-zero status (if it fails)
set -e

echo "Applying database migrations..."
alembic upgrade head

echo "Database migrations complete. Starting FastAPI server..."

# Use 'exec' so Uvicorn takes over the main process (PID 1). 
# This ensures Docker can send termination signals (like Ctrl+C) properly.
# NOTE: Change 'app.main:app' if your FastAPI instance is located elsewhere.
exec uvicorn app.api:app --host 0.0.0.0 --port 8000