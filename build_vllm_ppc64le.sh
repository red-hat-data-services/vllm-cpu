#!/bin/bash
set -eoux pipefail

# assume we are in vLLM's repo root
CURDIR=$(pwd)

# install development packages
rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm
microdnf install -y \
    git jq gcc-toolset-14 gcc-toolset-14-libatomic-devel automake libtool clang-devel openssl-devel freetype-devel fribidi-devel \
    harfbuzz-devel kmod lcms2-devel libimagequant-devel libjpeg-turbo-devel llvm15-devel \
    libraqm-devel libtiff-devel libwebp-devel libxcb-devel ninja-build openjpeg2-devel pkgconfig protobuf* \
    tcl-devel tk-devel xsimd-devel zeromq-devel zlib-devel file

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

source /opt/rh/gcc-toolset-14/enable
source /root/.cargo/env
export PATH=$PATH:/usr/lib64/llvm15/bin

export CMAKE_ARGS="-DPython3_EXECUTABLE=python"

uv pip install -U pip uv setuptools build wheel cmake

export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1

: ================== Installing Lapack ==================
# IMPORTANT: Ensure Lapack is installed in the final image
cd /root
export LAPACK_VERSION=${NUMACTL_VERSION:-$(curl -s https://api.github.com/repos/Reference-LAPACK/lapack/releases/latest | jq -r '.tag_name' | sed 's/v//')}
git clone --recursive https://github.com/Reference-LAPACK/lapack.git -b v${LAPACK_VERSION}
cd lapack
cmake -B build -S . && cmake --build build -j ${MAX_JOBS:-$(nproc)} && cmake --install build

: ================== Installing Numactl ==================
# IMPORTANT: Ensure Numactl is installed in the final image
cd /root
export NUMACTL_VERSION=${NUMACTL_VERSION:-$(curl -s https://api.github.com/repos/numactl/numactl/releases/latest | jq -r '.tag_name' | sed 's/v//')}
git clone --recursive https://github.com/numactl/numactl.git -b v${NUMACTL_VERSION}
cd numactl
autoreconf -i && ./configure && make -j ${MAX_JOBS:-$(nproc)} && make install

: ================== Installing OpenBlas ==================
# IMPORTANT: Ensure OpenBlas is installed in the final image
cd /root
export OPENBLAS_VERSION=${OPENBLAS_VERSION:-$(curl -s https://api.github.com/repos/OpenMathLib/OpenBLAS/releases/latest | jq -r '.tag_name' | sed 's/v//')}
curl -L https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz | tar xz
# rename directory for mounting (without knowing version numbers) in multistage builds
mv OpenBLAS-${OPENBLAS_VERSION}/ OpenBLAS/
cd OpenBLAS/
make -j${MAX_JOBS} TARGET=POWER9 BINARY=64 USE_OPENMP=1 USE_THREAD=1 NUM_THREADS=120 DYNAMIC_ARCH=1 INTERFACE64=0 && make install

# set path for openblas
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib/:/usr/local/lib64:/usr/local/lib
export PKG_CONFIG_PATH=$(find / -type d -name "pkgconfig" 2>/dev/null | tr '\n' ':')

cd ${CURDIR}


install_torch_family() {
    cd ${CURDIR}

    export TORCH_VERSION=${TORCH_VERSION:-$(grep -E '^torch==.+==\s*\"ppc64le\"' requirements/cpu.txt | grep -Eo '\b[0-9\.]+\b')}
    export TORCHVISION_VERSION=${TORCHVISION_VERSION:-$(grep -E '^torchvision==.+==\s*\"ppc64le\"' requirements/cpu.txt | grep -Eo '\b[0-9\.]+\b')}
    export TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-$(grep -E '^torchaudio==.+==\s*\"ppc64le\"' requirements/cpu.txt | grep -Eo '\b[0-9\.]+\b')}
    
    TEMP_BUILD_DIR=$(mktemp -d)
    cd ${TEMP_BUILD_DIR}
    
    : ================== Installing Pytorch ==================
    export _GLIBCXX_USE_CXX11_ABI=1
    git clone --recursive https://github.com/pytorch/pytorch.git -b v${TORCH_VERSION}
    cd pytorch
    sed -i '/lintrunner ;/s/$/ and platform_machine != "ppc64le"/' requirements.txt
    uv pip install -r requirements.txt
    python setup.py develop
    rm -f dist/torch*+git*whl
    MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    PYTORCH_BUILD_VERSION=${TORCH_VERSION} PYTORCH_BUILD_NUMBER=1 uv build --wheel --out-dir ${WHEEL_DIR}

    cd ${TEMP_BUILD_DIR}

    : ================== Installing Torchvision ==================
    export TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_FFMPEG=0
    git clone --recursive https://github.com/pytorch/vision.git -b v${TORCHVISION_VERSION}
    cd vision
    MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    BUILD_VERSION=${TORCHVISION_VERSION} \
    uv build --wheel --out-dir ${WHEEL_DIR} --no-build-isolation

    cd ${TEMP_BUILD_DIR}

    : ================== Installing Torchaudio ==================
    export BUILD_SOX=1 BUILD_KALDI=1 BUILD_RNNT=1 USE_FFMPEG=0 USE_ROCM=0 USE_CUDA=0
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_FFMPEG=1
    git clone --recursive https://github.com/pytorch/audio.git -b v${TORCHAUDIO_VERSION}
    cd audio
    MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    BUILD_VERSION=${TORCHAUDIO_VERSION} \
    uv build --wheel --out-dir ${WHEEL_DIR} --no-build-isolation

    cd ${CURDIR}
    rm -rf ${TEMP_BUILD_DIR}
}

install_pyarrow() {
    cd ${CURDIR}
    
    export PYARROW_VERSION=${PYARROW_VERSION:-$(curl -s https://api.github.com/repos/apache/arrow/releases/latest | jq -r '.tag_name' | grep -Eo "[0-9\.]+")}
    
    TEMP_BUILD_DIR=$(mktemp -d)
    cd ${TEMP_BUILD_DIR}
    
    : ================== Installing Pyarrow ==================
    git clone --recursive https://github.com/apache/arrow.git -b apache-arrow-${PYARROW_VERSION}
    cd arrow/cpp
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DARROW_PYTHON=ON \
        -DARROW_BUILD_TESTS=OFF \
        -DARROW_JEMALLOC=ON \
        -DARROW_BUILD_STATIC="OFF" \
        -DARROW_PARQUET=ON \
        ..
    make install -j ${MAX_JOBS:-$(nproc)}
    cd ../../python/
    uv pip install -v -r requirements-wheel-build.txt
    PYARROW_PARALLEL=${PYARROW_PARALLEL:-$(nproc)} \
    python setup.py build_ext \
        --build-type=release --bundle-arrow-cpp \
        bdist_wheel --dist-dir ${WHEEL_DIR}
    
    cd ${CURDIR}
    rm -rf ${TEMP_BUILD_DIR}
}

install_numba() {
    cd ${CURDIR}
    
    export NUMBA_VERSION=${NUMBA_VERSION:-$(grep -Eo '^numba.+;' requirements/cpu.txt | grep -Eo '\b[0-9\.]+\b' | tail -1)}
    
    TEMP_BUILD_DIR=$(mktemp -d)
    cd ${TEMP_BUILD_DIR}

    : ================== Installing Numba ==================
    git clone --recursive https://github.com/numba/numba.git -b ${NUMBA_VERSION}
    cd numba
    if ! grep '#include "dynamic_annotations.h"' numba/_dispatcher.cpp; then
        sed -i '/#include "internal\/pycore_atomic.h"/i\#include "dynamic_annotations.h"' numba/_dispatcher.cpp;
    fi
    
    export PATH=/usr/lib64/llvm15/bin:$PATH;
    export LLVM_CONFIG=/usr/lib64/llvm15/bin/llvm-config;
    
    uv build --wheel --out-dir ${WHEEL_DIR}

    cd ${CURDIR}
    rm -rf ${TEMP_BUILD_DIR}
}

install_torch_family
install_pyarrow
install_numba

#wait $(jobs -p)

# back to vLLM root
cd ${CURDIR}

# llvmlite==0.44.0 needs setuptools<70
echo "setuptools<70.0.0" > build_constraints.txt
uv pip install ${WHEEL_DIR}/numba*.whl --build-constraint build_constraints.txt

uv pip install ${WHEEL_DIR}/*.whl
sed -i.bak -e 's/.*torch.*//g' pyproject.toml requirements/*.txt
# sentencepiece.pc is in some pkgconfig inside uv cache
export PKG_CONFIG_PATH=$(find / -type d -name "pkgconfig" 2>/dev/null | tr '\n' ':')
uv pip install -r requirements/common.txt -r requirements/cpu.txt -r requirements/build.txt --index-strategy unsafe-best-match

