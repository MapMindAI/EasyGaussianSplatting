FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir open3d==0.19.0

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
