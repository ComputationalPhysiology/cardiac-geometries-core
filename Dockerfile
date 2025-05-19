from ubuntu:latest


# Dependency versions
ARG GMSH_VERSION=4_12_2

WORKDIR /tmp

# Install python3 and pip and gmsh
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    cmake \
    libeigen3-dev \
    ninja-build && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

ENV VENV=/opt/venv
RUN python3 -m venv $VENV \
    && $VENV/bin/pip install -U pip setuptools
ENV PATH="${VENV}/bin:$PATH"

# Install gmsh
RUN git clone -b gmsh_${GMSH_VERSION} --single-branch --depth 1 https://gitlab.onelab.info/gmsh/gmsh.git && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_BUILD_DYNAMIC=1 -B build-dir -S gmsh && \
    cmake --build build-dir && \
    cmake --install build-dir && \
    rm -rf /tmp/*

# GMSH installs python package in /usr/local/lib, see: https://gitlab.onelab.info/gmsh/gmsh/-/issues/1414
RUN export SP_DIR=$(python3 -c 'import sys, site; sys.stdout.write(site.getsitepackages()[0])') \
    && mv /usr/local/lib/gmsh.py ${SP_DIR}/ \
    && mv /usr/local/lib/gmsh*.dist-info ${SP_DIR}/

WORKDIR /app

COPY . /app

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install .

ENTRYPOINT ["geo"]
