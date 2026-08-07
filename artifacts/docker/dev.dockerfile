#FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM colmap/colmap:20260729.7651


SHELL [ "/bin/bash", "--login", "-c" ]

# Set locale.
RUN apt-get update -y && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Prepare and empty machine for building
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y \
    git \
    ninja-build \
    sudo \
    curl \
    wget \
    build-essential \
    unzip \
    gcc-10 \
    g++-10 \
    libvulkan-dev \
    vulkan-tools \
    # workaround: insta360_media_stitcher deadlocks on Vulkan device init failure
    # when a GPU is passed through with no ICD available; this gives it one
    mesa-vulkan-drivers \
    libglfw3-dev \
    libdc1394-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*


# installl insta360_sdk
COPY installers/insta360_sdk.sh /tmp/installers/
COPY installers/insta_360_main.cc /tmp/installers/
# COPY installers/libMediaSDK-dev.deb /tmp/installers/
RUN bash /tmp/installers/insta360_sdk.sh && rm /tmp/installers/insta360_sdk.sh

# installl install_exiftools
COPY installers/install_exiftools.sh /tmp/installers/
RUN bash /tmp/installers/install_exiftools.sh && rm /tmp/installers/install_exiftools.sh

# clean
RUN rm -rf /tmp/installers


# install miniconda
ENV CONDA_DIR /opt/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py312_24.5.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

COPY environment.yml /tmp/
RUN conda env create --file /tmp/environment.yml

RUN pip install jupyterlab

RUN conda activate gaussian_splatting
