# SNEPPX-ALG Docker Image
# Multi-stage build for minimal runtime images

# =====================================================================
# BASE BUILDER - Full development environment
# =====================================================================
FROM nvidia/cuda:13.5.0-devel-ubuntu22.04@sha256:b44fc069ef34cf454cdeabadf6a4e8b1d093b973b7d08ac2768bbc99fba2f013 AS builder

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
FROM nvidia/cuda:13.5.0-runtime-ubuntu22.04@sha256:a69a11a11011db340c0c4572e7decb923f4bbbcd129f2b3cae878d48eeb4536a AS cuda-runtime

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
FROM nvidia/cuda:13.5.0-devel-ubuntu22.04@sha256:b44fc069ef34cf454cdeabadf6a4e8b1d093b973b7d08ac2768bbc99fba2f013 AS dev

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
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS py-runtime

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
LABEL org.opencontainers.image.description="Experimental native C/C++ machine-learning framework with Python tensor, autograd, and neural-network APIs"
ARG SNEPPX_VERSION=1.5.3
LABEL org.opencontainers.image.version="${SNEPPX_VERSION}"
LABEL org.opencontainers.image.authors="Ammar [SNEPPX] <algoSNEPPX@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/SneppX/sneppx-alg"
LABEL org.opencontainers.image.documentation="https://sneppx.github.io/sneppx-alg/"
LABEL org.opencontainers.image.licenses="MIT"
