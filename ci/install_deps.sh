#!/usr/bin/env bash
set -eou pipefail

apt update && apt install -y git zlib1g-dev curl ninja-build

VERSION="3.16.5"
MIRROR_URL="https://github.com/Kitware/CMake/releases/download/v$VERSION"
DOWNLOAD="cmake-$VERSION-linux-x86_64.sh"
DOWNLOAD_FILE="/tmp/cmake.sh"

curl -Ls "${MIRROR_URL}/${DOWNLOAD}" --output "${DOWNLOAD_FILE}"
bash "${DOWNLOAD_FILE}" --skip-license --prefix=/usr/local --exclude-subdir
cmake --version
