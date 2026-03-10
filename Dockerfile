# 1. Use an official Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Set environment variables
# Prevents Python from writing .pyc files and ensures output is sent to logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install system dependencies (needed for some database drivers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code AND Alembic migration files
COPY ./app ./app
COPY ./storage ./storage
COPY ./alembic ./alembic
COPY alembic.ini .

# 7. Copy the startup script and make it executable
COPY start.sh .
RUN chmod +x start.sh

# 8. Expose the port FastAPI will run on
EXPOSE 8000

# 9. Command to run the application using the startup script
CMD ["./start.sh"]