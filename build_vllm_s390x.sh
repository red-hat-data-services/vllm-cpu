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
microdnf install -y \
    which procps findutils tar vim git gcc gcc-gfortran g++ gcc-c++ make patch zlib-devel \
    libjpeg-turbo-devel libtiff-devel libpng-devel libwebp-devel freetype-devel harfbuzz-devel \
    openssl-devel openblas openblas-devel autoconf automake libtool libzstd-devel cmake numpy libsndfile \
    clang clang-devel ninja-build perl-core llvm llvm-devel && \
    microdnf clean all

pip install --no-cache -U pip setuptools wheel && \
pip install --no-cache -U uv

curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    source "$CARGO_HOME/env" && \
    rustup default stable && \
    rustup show

# -------------------------
# Apache Arrow (C++ + Python)
# -------------------------

cd ${CURDIR}

#export PYARROW_VERSION=19.0.1
git clone --recursive https://github.com/apache/arrow.git
cd arrow/cpp

# Patch Arrow to disable xsimd includes (avoid version compatibility issues)
sed -i 's/#include <xsimd\/xsimd.hpp>/\/\/ #include <xsimd\/xsimd.hpp>/' \
    src/arrow/util/bpacking_simd128_generated_internal.h
sed -i 's/#include "arrow\/util\/bpacking_simd128_generated_internal.h"/\/\/ #include "arrow\/util\/bpacking_simd128_generated_internal.h"/' \
    src/arrow/util/bpacking_simd_default.cc

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
      -DARROW_USE_SIMD=OFF \
      -DARROW_SIMD_LEVEL=NONE \
      -DARROW_RUNTIME_SIMD_LEVEL=NONE \
      -DARROW_DEPENDENCY_SOURCE=BUNDLED \
      ..
make -j"$(nproc)"
make install
cd ../../python
export PYARROW_PARALLEL=4
export ARROW_BUILD_TYPE=release
uv pip install -r requirements-build.txt
python setup.py build_ext --build-type="$ARROW_BUILD_TYPE" --bundle-arrow-cpp bdist_wheel --dist-dir "${WHEEL_DIR}"

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
export TORCH_VERSION=2.8.0
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
# Build LLVM 15 from source
# -------------------------
cd ${CURDIR}
export MAX_JOBS=${MAX_JOBS:-"$(nproc)"}

echo "Building LLVM 15 from source..."
git clone --recursive https://github.com/llvm/llvm-project.git -b llvmorg-15.0.7
cd llvm-project
mkdir build
cd build

export PREFIX=/usr/local
export CMAKE_ARGS=""
CMAKE_ARGS="${CMAKE_ARGS} -DLLVM_ENABLE_PROJECTS=lld;libunwind;compiler-rt"
CFLAGS="$(echo ${CFLAGS:-} | sed 's/-fno-plt //g')"
CXXFLAGS="$(echo ${CXXFLAGS:-} | sed 's/-fno-plt //g')"
CMAKE_ARGS="${CMAKE_ARGS} -DFFI_INCLUDE_DIR=$PREFIX/include"
CMAKE_ARGS="${CMAKE_ARGS} -DFFI_LIBRARY_DIR=$PREFIX/lib"

cmake -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_PATH="${PREFIX}" \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_DOCS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_GO_TESTS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_UTILS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_UTILS_INSTALL_DIR=libexec/llvm \
    -DLLVM_BUILD_LLVM_DYLIB=OFF \
    -DLLVM_LINK_LLVM_DYLIB=OFF \
    -DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly \
    -DLLVM_ENABLE_FFI=ON \
    -DLLVM_ENABLE_Z3_SOLVER=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DCMAKE_POLICY_DEFAULT_CMP0111=NEW \
    -DCOMPILER_RT_BUILD_BUILTINS=ON \
    -DCOMPILER_RT_BUILTINS_HIDE_SYMBOLS=OFF \
    -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
    -DCOMPILER_RT_BUILD_CRT=OFF \
    -DCOMPILER_RT_BUILD_MEMPROF=OFF \
    -DCOMPILER_RT_BUILD_PROFILE=OFF \
    -DCOMPILER_RT_BUILD_SANITIZERS=OFF \
    -DCOMPILER_RT_BUILD_XRAY=OFF \
    -DCOMPILER_RT_BUILD_GWP_ASAN=OFF \
    -DCOMPILER_RT_BUILD_ORC=OFF \
    -DCOMPILER_RT_INCLUDE_TESTS=OFF \
    ${CMAKE_ARGS} -GNinja ../llvm

ninja install
echo "LLVM 15 build complete."

# Update PATH and LD_LIBRARY_PATH for LLVM
export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# -------------------------
# numba & llvmlite
# -------------------------
export MAX_JOBS=${MAX_JOBS:-"$(nproc)"}
export NUMBA_VERSION=0.61.2
cd ${CURDIR}

# Set up LLVM environment variables before building llvmlite
export LLVM_CONFIG=/usr/local/bin/llvm-config
export LLVM_PREFIX=/usr/local
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

# Install llvmlite before building numba (numba depends on llvmlite)
uv pip install "${WHEEL_DIR}"/llvmlite*.whl

cd ${CURDIR}
cd numba
if ! grep -F '#include "dynamic_annotations.h"' numba/_dispatcher.cpp; then
   sed -i '/#include "internal\/pycore_atomic.h"/i #include "dynamic_annotations.h"' numba/_dispatcher.cpp
fi
python setup.py bdist_wheel --dist-dir "${WHEEL_DIR}"

# -------------------------
# outlines-core
# -------------------------
cd ${CURDIR}
export OUTLINES_CORE_VERSION=0.2.11
git clone https://github.com/dottxt-ai/outlines-core.git
cd outlines-core
git checkout tags/${OUTLINES_CORE_VERSION}
sed -i 's/version = "0.0.0"/version = "'"${OUTLINES_CORE_VERSION}"'"/' Cargo.toml
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
