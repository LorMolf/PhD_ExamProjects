# Use a Python base image
FROM python:3.8-slim
LABEL maintainer="disi-unibo-nlp"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app


# Update and install dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        bash \
        nano \
        wget \
        libgomp1 \
        build-essential \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir Cython==3.0.11 \
                                tqdm==4.67.1 \
                                matplotlib==3.7.5 \
                                seaborn==0.13.2 \
                                numpy==1.24.4

ENV DEBIAN_FRONTEND=dialog