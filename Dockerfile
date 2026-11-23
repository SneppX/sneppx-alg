# SNEPPX-ALG Docker Image
# Multi-stage build for minimal runtime images

# =====================================================================
# BASE BUILDER - Full development environment
# =====================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04@sha256:ae8a022c02aec945c4f8c52f65deaf535de7abb58e840350d19391ec683f4980 AS builder

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
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04@sha256:94c1577b2cd9dd6c0312dc04dff9cb2fdce2b268018abc3d7c2dbcacf1155000 AS cuda-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libopenblas0 \
    libomp5 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r sneppx && useradd -r -g sneppx sneppx

# Copy installed artifacts from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /install /usr/local

# Install Python package
COPY --from=builder /install/lib/${PYTHON_SITE}/dist-packages /usr/local/lib/${PYTHON_SITE}/dist-packages

# Set Python path
ENV PYTHONPATH=/usr/local/lib/${PYTHON_SITE}/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN mkdir -p /workspace && chown sneppx:sneppx /workspace
WORKDIR /workspace
USER sneppx

# =====================================================================
# CPU-ONLY RUNTIME IMAGE
# =====================================================================
FROM ubuntu:22.04@sha256:f64b2ffc2f10ab8f8505aa56b2664bd846cbb9df87c7e2532e30f7667021c0e6 AS cpu-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libopenblas0 \
    libomp5 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r sneppx && useradd -r -g sneppx sneppx

COPY --from=builder /usr/local /usr/local
COPY --from=builder /install /usr/local

# Install Python package
COPY --from=builder /install/lib/${PYTHON_SITE}/dist-packages /usr/local/lib/${PYTHON_SITE}/dist-packages

ENV PYTHONPATH=/usr/local/lib/${PYTHON_SITE}/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN mkdir -p /workspace && chown sneppx:sneppx /workspace
WORKDIR /workspace
USER sneppx

# =====================================================================
# DEVELOPMENT IMAGE (Full CUDA + dev tools)
# =====================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04@sha256:ae8a022c02aec945c4f8c52f65deaf535de7abb58e840350d19391ec683f4980 AS dev

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
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS py-runtime

ARG PYTHON_SITE=python3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r sneppx && useradd -r -g sneppx sneppx

COPY --from=builder /usr/local/lib/${PYTHON_SITE}/dist-packages /usr/local/lib/${PYTHON_SITE}/dist-packages
COPY --from=builder /usr/local/lib /usr/local/lib

RUN pip3 install --no-cache-dir uvicorn==0.54.0

ENV PYTHONPATH=/usr/local/lib/${PYTHON_SITE}/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

RUN mkdir -p /workspace && chown sneppx:sneppx /workspace
WORKDIR /workspace
USER sneppx

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
