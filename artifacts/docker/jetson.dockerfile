# gsplat training server for the Jetson AGX Orin (JetPack 6, L4T r36.4, CUDA
# 12.6). arm64 only. The image includes COLMAP for inspection and repair; the
# service itself trains uploaded perspective models with gsplat.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
SHELL [ "/bin/bash", "-c" ]

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    python3-dev \
    protobuf-compiler \
    protobuf-compiler-grpc \
    python3-pip \
    unzip \
    cmake \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libceres-dev \
    libeigen3-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libopenimageio-dev \
    libsqlite3-dev \
    libssl-dev \
    libomp-dev \
    libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Sourced from the third_party/gsplat submodule via the "gsplatsrc" build
# context (see README for the --build-context flag this requires), not the
# default build context, so the image build doesn't have to send the whole repo.
COPY --from=gsplatsrc . /opt/gsplat
COPY --from=reposrc gsplat_server/config/gsplat_train_defaults.proto.txt scripts/gsplat_env.sh /opt/easygaussiansplatting/scripts/
COPY --from=reposrc mapping/train_gsplat_with_masks.py /opt/easygaussiansplatting/mapping/
COPY --from=reposrc gsplat_server /opt/easygaussiansplatting/gsplat_server
COPY --from=colmapsrc . /tmp/colmap
RUN mkdir -p /usr/include/opencv4 \
    && cmake -S /tmp/colmap -B /tmp/colmap/build -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DCOLMAP_HASH_MAP_BACKEND=STD -DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF \
        -DMVS_ENABLED=OFF -DONNX_ENABLED=OFF -DCGAL_ENABLED=OFF -DTESTS_ENABLED=OFF \
    && cmake --build /tmp/colmap/build --parallel \
    && cmake --install /tmp/colmap/build \
    && rm -rf /tmp/colmap
COPY installers/install_gsplat_orin.sh /tmp/
RUN bash /tmp/install_gsplat_orin.sh && rm /tmp/install_gsplat_orin.sh
RUN bash /opt/easygaussiansplatting/gsplat_server/proto/build.sh

WORKDIR /opt/easygaussiansplatting

EXPOSE 50051
