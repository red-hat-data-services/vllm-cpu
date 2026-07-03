#!/bin/bash
set -eoux pipefail

########################################
# Resolve repo root (IMPORTANT)
########################################
REPO_ROOT="$(pwd)"

cd "$REPO_ROOT"

########################################
# DevPI configuration
########################################

IBM_DEVPI_URL=${IBM_DEVPI_URL:-"https://wheels.developerfirst.ibm.com/ppc64le/linux/+simple/"}
RHOAI_INDEX_URL=${RHOAI_INDEX_URL:-"https://packages.redhat.com/api/pypi/public-rhai/rhoai/3.5-EA2/cpu-ubi9/simple/"}

########################################
# wheel dir
########################################

mkdir -p $WHEEL_DIR

########################################
# Helpers
########################################
try_install_from_devpi() {
    local pkg=$1
    uv pip install \
        --extra-index-url "${IBM_DEVPI_URL}" \
        --index-strategy unsafe-best-match \
        --no-build-isolation \
        "${pkg}"
}

try_install_from_rhoai() {
    local pkg=$1
    uv pip install \
        --extra-index-url "${RHOAI_INDEX_URL}" \
        --index-strategy unsafe-best-match \
        --no-build-isolation \
        "${pkg}"
}


########################################
# install system dependencies
########################################

rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm || true

microdnf install -y \
    python3.12 python3.12-devel python3.12-pip gcc \
    git jq gcc-toolset-14 gcc-toolset-14-libatomic-devel \
    automake libtool clang-devel openssl-devel \
    harfbuzz-devel kmod lcms2-devel libimagequant-devel libjpeg-turbo-devel \
    llvm15-devel libraqm-devel libtiff-devel libwebp-devel libxcb-devel \
    ninja-build openjpeg2-devel pkgconfig \
    tcl-devel tk-devel xsimd-devel zeromq-devel zlib-devel patchelf file openblas openblas-devel protobuf numactl numactl-devel openmpi openmpi-devel

########################################
# Python 3.12 virtual environment
########################################

python3.12 -m venv /opt/vllm
source /opt/vllm/bin/activate

export PATH=/opt/vllm/bin:$PATH

python --version

########################################
# install build tools (stable uv)
########################################

pip install -U pip setuptools-rust
pip install uv
pip install "setuptools<70" build wheel cmake auditwheel
uv pip install "setuptools<70" cython meson-python pybind11 "sympy>=1.13.3" --no-build-isolation

########################################
# Rust
########################################

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source /root/.cargo/env

########################################
# Compiler env
########################################

source /opt/rh/gcc-toolset-14/enable

export PATH=/usr/lib64/llvm15/bin:$PATH
export LLVM_CONFIG=/usr/lib64/llvm15/bin/llvm-config
export CMAKE_ARGS="-DPython3_EXECUTABLE=python"

export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1

########################################
# PROTOBUF
########################################
cd /root
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v25.8
git submodule update --init --recursive
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dprotobuf_BUILD_TESTS=OFF \
  -Dprotobuf_BUILD_SHARED_LIBS=ON \
  -Dprotobuf_ABSL_PROVIDER=module \
  -DABSL_ENABLE_INSTALL=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_FLAGS="-O2 -fPIC -mcmodel=medium" \
  -DCMAKE_INSTALL_PREFIX=/usr/local

make -j$(nproc)
make install
ldconfig

export CMAKE_PREFIX_PATH=/usr/local:${CMAKE_PREFIX_PATH:-}
export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}


########################################
# DevPI Packages
########################################
uv pip install numpy==2.3.5 pillow==12.2.0 --extra-index-url "$IBM_DEVPI_URL"
install_packages() {
    try_install_from_devpi "opencv-python-headless==${OPENCV_VERSION}"
    try_install_from_devpi "torch==${TORCH_VERSION}"
    try_install_from_devpi "torchvision==${TORCHVISION_VERSION}"
    try_install_from_rhoai "torchaudio==${TORCHAUDIO_VERSION}"

    ### RHOAI INDEX ###
    pip download \
        --index-url "${RHOAI_INDEX_URL}" \
        --only-binary=:all: \
        --no-deps \
        llvmlite==0.47.0 \
        -d ${WHEEL_DIR}

    pip download \
        --index-url "${RHOAI_INDEX_URL}" \
        --only-binary=:all: \
        --no-deps \
        xgrammar==0.2.1 \
        -d ${WHEEL_DIR}
}

########################################
# Package Versions
########################################
cd "$REPO_ROOT"
TORCH_VERSION=${TORCH_VERSION:-$(grep -E '^torch==.+==\s*"ppc64le"' requirements/cpu.txt | grep -Eo '\b[0-9\.]+\b')}
TORCH_VERSION=${TORCH_VERSION:-2.11.0}

TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.26.0}
TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-${TORCH_VERSION}}

export TORCH_VERSION
export TORCHVISION_VERSION
export TORCHAUDIO_VERSION
export OPENCV_VERSION=${OPENCV_VERSION:-4.13.0.92}

########################################
# RUN BUILDS
########################################

install_packages

########################################
# install built wheels
########################################
uv pip install setuptools_scm maturin setuptools-rust ninja scikit-build-core pybind11 nanobind \
    --no-build-isolation
uv pip install ${WHEEL_DIR}/*.whl \
    --extra-index-url "$IBM_DEVPI_URL" \
    --index-strategy unsafe-best-match \
    --no-build-isolation

########################################
# install remaining deps
########################################

sed -i.bak -e 's/.*torch.*//g' pyproject.toml requirements/*.txt
sed -i '/fastapi\[standard\]/ s/>= 0\.120\.1/>= 0.120.1, < 0.137/' requirements/common.txt
# revert back for numba/llvmlite compatibility
uv pip install "setuptools<70" --no-build-isolation

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig:/usr/lib64/pkgconfig

uv pip install -r requirements/common.txt \
               -r requirements/cpu.txt \
               -r requirements/build/cpu.txt --extra-index-url "$IBM_DEVPI_URL" --index-strategy unsafe-best-match
