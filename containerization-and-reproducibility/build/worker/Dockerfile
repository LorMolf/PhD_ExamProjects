FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
LABEL maintainer="disi-unibo-nlp"

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.9 \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the worker requirements into the container
COPY build/worker/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the worker script
COPY src/train.py .

CMD ["python3", "train.py"]
