#!/usr/bin/env bash

set -e

mkdir -p /tmp/installers
pushd /tmp/installers

# https://exiftool.org/index.html
EXIFTOOL_VERSION=13.59
wget -O Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz https://sourceforge.net/projects/exiftool/files/Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz/download
gzip -dc Image-ExifTool-${EXIFTOOL_VERSION}.tar.gz | tar -xf -
cd Image-ExifTool-${EXIFTOOL_VERSION}
perl Makefile.PL
make test
make install

popd
