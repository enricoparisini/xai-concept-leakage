FROM python:3.7-slim

# set non-interactive frontend for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip and install pip tools
RUN pip install --upgrade pip setuptools wheel

# cd to home
WORKDIR /home/xai-concept-leakage

# copy the rest of the source and install it 
COPY . .
RUN python -m pip install --no-cache-dir .

# set working directory
WORKDIR /home

# default command for container
CMD ["/bin/bash"]







