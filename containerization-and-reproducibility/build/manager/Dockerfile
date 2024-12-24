FROM python:3.9-slim

WORKDIR /app

# Copy manager requirements
COPY build/manager/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the manager script from src
COPY src/manager.py .

# We'll rely on docker-compose to set the startup command ("python manager.py").
