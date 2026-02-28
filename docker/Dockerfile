FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (includes capnp headers for pycapnp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libcapnp-dev \
    capnproto \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip for PEP 621 (pyproject.toml) support, then install deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Julia install via juliaup (installs to ~/.juliaup by default)
ENV JULIA_DEPOT_PATH=/usr/local/julia-depot
ENV PATH="/root/.juliaup/bin:${PATH}"
RUN curl -fsSL https://install.julialang.org | sh -s -- -y --default-channel 1.10

# Julia packages (changes rarely — slow layer, cached)
COPY training/install_packages.jl training/install_packages.jl
RUN julia training/install_packages.jl

# Application code (changes often — fast rebuild)
COPY . .

# Install nnlc_tools package (with dev deps for testing)
RUN pip install --no-cache-dir ".[dev]"

# Volume mount points
VOLUME ["/app/data", "/app/output"]

CMD ["bash"]
