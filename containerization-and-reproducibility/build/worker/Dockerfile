FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
LABEL maintainer="disi-unibo-nlp"

# Set working directory
WORKDIR /app

# Update and install dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        bash \
        nano \
        wget \
        python3.9 \
        libgomp1 \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Ensure NVIDIA runtime support (delegated to docker-compose.yml)
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Copy and install Python dependencies
COPY build/worker/requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the worker script
COPY src/train.py /app/

# Set default command
CMD ["python3", "train.py"]
