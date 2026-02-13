FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-numpy python3-opencv \
    v4l-utils \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir v4l2-fix jetson-stats

WORKDIR /work