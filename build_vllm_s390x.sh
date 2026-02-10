#!/bin/bash
set -eoux pipefail

# Set up environment variables
export PYTHON_VERSION=3.12
export WHEEL_DIR=/wheelsdir
export HOME=/root
export CURDIR=$(pwd)
export VIRTUAL_ENV=/opt/venv
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export PATH=$CARGO_HOME/bin:$RUSTUP_HOME/bin:$PATH
export C_INCLUDE_PATH=${C_INCLUDE_PATH:-/usr/local/include}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/usr/lib64:/usr/lib}

# install development packages
rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
microdnf install -y \
    which procps findutils tar vim git gcc-toolset-14 gcc-toolset-14-binutils gcc-toolset-14-libatomic-devel  make patch zlib-devel \
    libjpeg-turbo-devel libtiff-devel libpng-devel libwebp-devel freetype-devel harfbuzz-devel \
    openssl-devel openblas openblas-devel autoconf automake libtool cmake numpy libsndfile \
    clang clang-devel  ninja-build perl-core libsodium libsodium-devel llvm15 llvm15-devel llvm15-static && \
    microdnf clean all

pip install --no-cache -U pip wheel && \
pip install --no-cache -U uv

source /opt/rh/gcc-toolset-14/enable

curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    source "$CARGO_HOME/env" && \
    rustup default stable && \
    rustup show

# -------------------------
# Apache Arrow (C++ + Python)
# -------------------------

cd ${CURDIR}

git clone https://github.com/apache/arrow.git
cd arrow/cpp
mkdir -p release
cd release
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DARROW_PYTHON=ON \
      -DARROW_PARQUET=ON \
      -DARROW_ORC=ON \
      -DARROW_FILESYSTEM=ON \
      -DARROW_WITH_LZ4=ON \
      -DARROW_WITH_ZSTD=ON \
      -DARROW_WITH_SNAPPY=ON \
      -DARROW_JSON=ON \
      -DARROW_CSV=ON \
      -DARROW_DATASET=ON \
      -DPROTOBUF_PROTOC_EXECUTABLE=/usr/bin/protoc \
      -DARROW_DEPENDENCY_SOURCE=BUNDLED \
      ..
make -j"$(nproc)"
make install
cd ../../python
export PYARROW_PARALLEL=4
export ARROW_BUILD_TYPE=release
uv pip install -r requirements-build.txt
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
python setup.py build_ext --build-type=$ARROW_BUILD_TYPE --bundle-arrow-cpp --inplace
python setup.py bdist_wheel --dist-dir "${WHEEL_DIR}"

# -------------------------
# numactl
# -------------------------
cd ${CURDIR}
curl -LO https://github.com/numactl/numactl/archive/refs/tags/v2.0.18.tar.gz
tar -xvzf v2.0.18.tar.gz
mv numactl-2.0.18/ numactl/
cd numactl
./autogen.sh
./configure
make
make install
export C_INCLUDE_PATH="/usr/local/include:$C_INCLUDE_PATH"

# -------------------------
# PyTorch
# -------------------------
export TORCH_VERSION=2.7.0
export _GLIBCXX_USE_CXX11_ABI=1
export CARGO_HOME=/root/.cargo
export RUSTUP_HOME=/root/.rustup
export PATH="$CARGO_HOME/bin:$RUSTUP_HOME/bin:$PATH"
cd ${CURDIR}
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v${TORCH_VERSION}
git submodule sync
git submodule update --init --recursive
uv pip install cmake ninja
uv pip install -r requirements.txt
PYTORCH_BUILD_VERSION=${TORCH_VERSION} PYTORCH_BUILD_NUMBER=1 uv build --wheel --out-dir ${WHEEL_DIR}
uv pip install ${WHEEL_DIR}/torch-${TORCH_VERSION}*.whl

# -------------------------
# TorchVision
# -------------------------
export TORCH_VISION_VERSION=v0.20.1
cd ${CURDIR}
uv pip install cmake ninja wheel setuptools
git clone https://github.com/pytorch/vision.git
cd vision
git checkout $TORCH_VISION_VERSION
MAX_JOBS=${MAX_JOBS:-$(nproc)}
uv build --wheel --out-dir ${WHEEL_DIR} --no-build-isolation

# -------------------------
# hf-xet
# -------------------------
cd ${CURDIR}
git clone https://github.com/huggingface/xet-core.git
cd xet-core/hf_xet/
uv pip install maturin patchelf
python -m maturin build --release --out "${WHEEL_DIR}"

# -------------------------
# numba & llvmlite
# -------------------------
export MAX_JOBS=${MAX_JOBS:-"$(nproc)"}
export NUMBA_VERSION=0.61.2
cd ${CURDIR}

# Set up LLVM environment variables before building llvmlite
export LLVM_CONFIG=/usr/bin/llvm-config-15
export LLVM_PREFIX=/usr
export CFLAGS=""
export CXXFLAGS=""

# Clone and build llvmlite and numba
git clone --recursive https://github.com/numba/llvmlite.git -b v0.44.0
git clone --recursive https://github.com/numba/numba.git -b ${NUMBA_VERSION}

uv pip install 'cmake<4' setuptools numpy
cd ${CURDIR}

# Now build llvmlite with the installed LLVM
cd llvmlite
python setup.py bdist_wheel --dist-dir "${WHEEL_DIR}"

cd ${CURDIR}
cd numba
if ! grep -F '#include "dynamic_annotations.h"' numba/_dispatcher.cpp; then
   sed -i '/#include "internal\/pycore_atomic.h"/i #include "dynamic_annotations.h"' numba/_dispatcher.cpp
fi
python setup.py bdist_wheel --dist-dir "${WHEEL_DIR}"

# -------------------------
# aws-lc-sys patch (s390x)
# -------------------------
cd ${CURDIR}

export AWS_LC_VERSION=v0.32.0
git clone --recursive https://github.com/aws/aws-lc-rs.git
cd aws-lc-rs
git checkout tags/aws-lc-sys/${AWS_LC_VERSION}
git submodule sync
git submodule update --init --recursive
cd aws-lc-sys
sed -i '682 s/strncmp(buf, \"-----END \", 9)/memcmp(buf, \"-----END \", 9)/' aws-lc/crypto/pem/pem_lib.c
sed -i '712 s/strncmp(buf, \"-----END \", 9)/memcmp(buf, \"-----END \", 9)/' aws-lc/crypto/pem/pem_lib.c
sed -i '747 s/strncmp(buf, \"-----END \", 9)/memcmp(buf, \"-----END \", 9)/' aws-lc/crypto/pem/pem_lib.c

# -------------------------
# outlines-core (patched to local aws-lc-sys)
# -------------------------
cd ${CURDIR}
export OUTLINES_CORE_VERSION=0.2.10
git clone https://github.com/dottxt-ai/outlines-core.git
cd outlines-core
git checkout tags/${OUTLINES_CORE_VERSION}
sed -i 's/version = "0.0.0"/version = "'"${OUTLINES_CORE_VERSION}"'"/' Cargo.toml
echo '[patch.crates-io]' >> Cargo.toml
echo 'aws-lc-sys = { path = "/root/aws-lc-rs/aws-lc-sys" }' >> Cargo.toml
uv pip install maturin
python -m maturin build --release --out "${WHEEL_DIR}"

# -------------------------
# install all wheels we've built so far
# -------------------------
cd ${CURDIR}

mkdir -p lapack
mkdir -p OpenBLAS

uv pip install ${WHEEL_DIR}/*.whl

sed -i.bak -e 's/.*torch.*//g' pyproject.toml requirements/*.txt
export PKG_CONFIG_PATH=$(find / -type d -name "pkgconfig" 2>/dev/null | tr '\n' ':')
uv pip install -r requirements/common.txt -r requirements/cpu.txt -r requirements/build.txt --index-strategy unsafe-best-match
