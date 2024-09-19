# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies for OpenFHE and JupyterLab
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyBind11
RUN pip3 install "pybind11[global]"

# Clone and build OpenFHE-development version 1.2.0
RUN git clone --depth 1 --branch v1.2.0 https://github.com/openfheorg/openfhe-development.git \
    && cd openfhe-development \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF .. \
    && make -j$(nproc) \
    && make install

# Assume that OpenFHE installs libraries into /usr/local/lib
# Update LD_LIBRARY_PATH to include this directory
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Clone and build OpenFHE-Python version 0.8.8
RUN git clone --depth 1 --branch v0.8.8 https://github.com/openfheorg/openfhe-python.git \
    && cd openfhe-python \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j$(nproc) \
    && make install
