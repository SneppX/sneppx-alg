# SNEPPX-ALG Docker Image
# Multi-stage build for minimal runtime images

# =====================================================================
# BASE BUILDER - Full development environment
# =====================================================================
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS builder

ARG PYTHON_SITE=python3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build \
    build-essential \
    python3 python3-pip python3-dev python3-venv \
    git \
    wget curl \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copy source code
WORKDIR /sneppx
COPY . .

# Build CMake project
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSNEPPX_BUILD_TESTS=ON \
    -DSNEPPX_BUILD_PYTHON=ON \
    -DSNEPPX_BUILD_CUDA=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build -j$(nproc) && \
    cmake --install build --prefix /install

# Install Python package
RUN pip3 install --no-cache-dir ./bindings/python

# =====================================================================
# CUDA RUNTIME IMAGE
# =====================================================================
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS cuda-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libopenblas0 \
    libomp5 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed artifacts from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /install /usr/local

# Install Python package
COPY --from=builder /install/lib/${PYTHON_SITE}/dist-packages /usr/local/lib/${PYTHON_SITE}/dist-packages

# Set Python path
ENV PYTHONPATH=/usr/local/lib/${PYTHON_SITE}/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

# =====================================================================
# CPU-ONLY RUNTIME IMAGE
# =====================================================================
FROM ubuntu:22.04 AS cpu-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libopenblas0 \
    libomp5 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /install /usr/local

# Install Python package
COPY --from=builder /install/lib/${PYTHON_SITE}/dist-packages /usr/local/lib/${PYTHON_SITE}/dist-packages

ENV PYTHONPATH=/usr/local/lib/${PYTHON_SITE}/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

# =====================================================================
# DEVELOPMENT IMAGE (Full CUDA + dev tools)
# =====================================================================
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS dev

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build \
    build-essential \
    python3 python3-pip python3-dev python3-venv \
    git \
    wget curl \
    libopenblas-dev \
    libomp-dev \
    vim tmux htop \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /sneppx
COPY . .

# Install in development mode
RUN pip install -e ./bindings/python

WORKDIR /sneppx

# =====================================================================
# PYTHON-ONLY RUNTIME (Minimal)
# =====================================================================
FROM python:3.11-slim AS py-runtime

ARG PYTHON_SITE=python3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/${PYTHON_SITE}/dist-packages /usr/local/lib/${PYTHON_SITE}/dist-packages
COPY --from=builder /usr/local/lib /usr/local/lib

ENV PYTHONPATH=/usr/local/lib/${PYTHON_SITE}/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

# =====================================================================
# DEFAULT TARGET
# =====================================================================
FROM cuda-runtime AS default

LABEL org.opencontainers.image.title="SNEPPX-ALG"
LABEL org.opencontainers.image.description="Next-generation AI architecture with security built into the foundation"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.authors="Ammar [SNEPPX] <algoSNEPPX@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/ammar49-cyber/sneppx-alg"
LABEL org.opencontainers.image.licenses="MIT"