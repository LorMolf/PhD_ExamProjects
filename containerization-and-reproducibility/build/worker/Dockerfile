FROM python:3.9-slim

WORKDIR /app

# Copy worker requirements
COPY build/worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script from src
COPY src/train.py .

# We rely on docker-compose for "python train.py" as the entrypoint.
