#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

SDK_NAME="libMediaSDK-dev.deb"

if [ -f "${SDK_NAME}" ]; then
  echo "File exists"
else
  curl -L -o ${SDK_NAME} \
    https://github.com/MapMindAI/EasyGaussianSplatting/releases/download/v1.1/Insta360SDK.deb
fi

apt-get install -y ./${SDK_NAME}

# rm -rf /var/lib/apt/lists/*
g++ insta_360_main.cc -std=c++11 -lMediaSDK -lpthread  -o /usr/local/bin/insta360_media_stitcher
echo "Build done!"

# apt-get remove libMediaSDK-dev

rm -rf /var/lib/apt/lists/*

popd
