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

# Default startup command
CMD ["/bin/bash"]