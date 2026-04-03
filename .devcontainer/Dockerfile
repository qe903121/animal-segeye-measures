# Use the lightweight official Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and force unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies (OpenCV runtime libs and build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy and install AI dependencies
COPY requirements-ai.txt /tmp/
RUN pip install "numpy<2.0.0" cython "setuptools<70.0.0" && \
    pip install chumpy --no-build-isolation && \
    pip install --no-build-isolation --no-binary xtcocotools xtcocotools && \
    pip install --no-cache-dir -r /tmp/requirements-ai.txt && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html

# Default startup command
CMD ["/bin/bash"]