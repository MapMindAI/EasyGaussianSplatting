#FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM colmap/colmap:20260729.7651

# insta360_media_stitcher needs NVENC/NVDEC (libnvidia-encode/libnvcuvid), which the
# container runtime only mounts when the "video" capability is requested; the base
# image's "compute,utility" default leaves hardware encode failing with EPERM.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

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

# opencv-python-headless is for scripts/run_pipeline.sh and mapping/, which only
# need cv2 (no GUI) and so share the base env rather than a dedicated one.
RUN pip install --no-cache-dir jupyterlab opencv-python-headless

# gsplat (Gaussian Splatting training), in its own conda env: see
# installers/install_gsplat.sh for why it needs a different Python/CUDA
# stack than the base env above. Sourced from the third_party/gsplat
# submodule via the "gsplatsrc" build context (see README for the
# --build-context flag this requires), not the default build context, so
# the image build doesn't have to send the whole repo (including data/).
COPY --from=gsplatsrc . /opt/gsplat
COPY installers/install_gsplat.sh /tmp/installers/
RUN bash /tmp/installers/install_gsplat.sh && rm /tmp/installers/install_gsplat.sh

# pycolmap is pinned to the same version as the base image's COLMAP CLI.
# tritonclient talks to the inference server the mapping stage extracts and
# matches features against; matplotlib is imported by the SALAD client in
# third_party/EasyTensorRT for its own plotting entrypoint.
RUN pip install --no-cache-dir pycolmap==4.2.0.dev0 "tritonclient[grpc]" matplotlib
