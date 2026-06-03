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

if [[ -n "$IBM_DEVPI_URL" ]]; then
    echo "Using IBM's Python index: $IBM_DEVPI_URL"
    export PIP_EXTRA_INDEX_URL=$IBM_DEVPI_URL
fi

########################################
# install system dependencies
########################################

rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm || true

microdnf install -y \
    python3.12 python3.12-devel python3.12-pip gcc \
    git jq gcc-toolset-14 gcc-toolset-14-libatomic-devel \
    automake libtool clang-devel openssl-devel freetype-devel fribidi-devel \
    harfbuzz-devel kmod lcms2-devel libimagequant-devel libjpeg-turbo-devel \
    llvm15-devel libraqm-devel libtiff-devel libwebp-devel libxcb-devel \
    ninja-build openjpeg2-devel pkgconfig \
    tcl-devel tk-devel xsimd-devel zeromq-devel zlib-devel patchelf file

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
uv pip install "setuptools<70" cython meson-python --no-build-isolation

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
# wheel dir
########################################

mkdir -p $WHEEL_DIR

########################################
# DevPI helper
########################################

try_install_from_devpi() {
    pkg=$1
    if [[ -n "$IBM_DEVPI_URL" ]]; then
        if uv pip install \
            --extra-index-url "$IBM_DEVPI_URL" \
            --index-strategy unsafe-best-match \
            --no-build-isolation \
            "$pkg"; then
            echo "Installed $pkg from DevPI"
            return 0
        fi
    fi
    return 1
}

########################################
# LAPACK
########################################

cd /root
LAPACK_VERSION=$(curl -s https://api.github.com/repos/Reference-LAPACK/lapack/releases/latest | jq -r '.tag_name' | sed 's/v//')
git clone --depth 1 https://github.com/Reference-LAPACK/lapack.git -b v${LAPACK_VERSION}
cd lapack
cmake -B build -S .
cmake --build build -j ${MAX_JOBS}
cmake --install build

########################################
# NUMACTL
########################################

cd /root
NUMACTL_VERSION=$(curl -s https://api.github.com/repos/numactl/numactl/releases/latest | jq -r '.tag_name' | sed 's/v//')

git clone --depth 1 https://github.com/numactl/numactl.git -b v${NUMACTL_VERSION}
cd numactl

autoreconf -i
./configure
make -j ${MAX_JOBS}
make install

########################################
# OPENBLAS
########################################

cd /root
OPENBLAS_VERSION=$(curl -s https://api.github.com/repos/OpenMathLib/OpenBLAS/releases/latest | jq -r '.tag_name' | sed 's/v//')

curl -L https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz | tar xz

mv OpenBLAS-${OPENBLAS_VERSION}/ OpenBLAS/
cd OpenBLAS/

make -j${MAX_JOBS} TARGET=POWER9 BUILD_BFLOAT16=1 BINARY=64 USE_OPENMP=1 USE_THREAD=1 NUM_THREADS=120 DYNAMIC_ARCH=1 INTERFACE64=0
make install

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib

########################################
# PROTOBUF
########################################

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
# ABSEIL CPP
########################################

# git clone https://github.com/abseil/abseil-cpp.git
# cd abseil-cpp
# mkdir build && cd build

# cmake .. \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
#   -DCMAKE_CXX_STANDARD=17 \
#   -DCMAKE_INSTALL_PREFIX=/usr/local \
#   -DCMAKE_CXX_FLAGS="-fPIC -mcmodel=medium"

# make -j$(nproc)
# make install

########################################
# PYTORCH FAMILY
########################################

install_torch_family() {

    cd "$REPO_ROOT"
    TORCH_VERSION=${TORCH_VERSION:-$(grep -E '^torch==.+==\s*"ppc64le"' requirements/cpu.txt | grep -Eo '\b[0-9\.]+\b')}
    TORCH_VERSION=${TORCH_VERSION:-2.11.0}
    export TORCH_VERSION

    TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.26.0}
    export TORCHVISION_VERSION

    TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION:-$TORCH_VERSION}
    export TORCHAUDIO_VERSION

    echo "Torch version      : $TORCH_VERSION"
    echo "Torchvision version: $TORCHVISION_VERSION"
    echo "Torchaudio version : $TORCHAUDIO_VERSION"

    echo "========== Installing SymPy =========="
    uv pip install \
        --extra-index-url "$IBM_DEVPI_URL" \
        --index-strategy unsafe-best-match \
        "sympy>=1.13.3"

    echo "========== Installing Torch from DevPI =========="
    try_install_from_devpi "torch==${TORCH_VERSION}"

    echo "========== Installing Torchvision from DevPI =========="
    try_install_from_devpi "torchvision==${TORCHVISION_VERSION}"

    echo "Torch and Torchvision installed from DevPI"

    TEMP_BUILD_DIR=$(mktemp -d)
    cd ${TEMP_BUILD_DIR}

    export BLAS=OpenBLAS
    export USE_OPENMP=1
    export USE_MKLDNN=OFF
    export USE_MKLDNN_CBLAS=OFF
    export OPENBLAS_HOME="/usr/local"
    export PKG_CONFIG_PATH="$OPENBLAS_HOME/lib/pkgconfig:${PKG_CONFIG_PATH}"
    export LIBRARY_PATH="$OPENBLAS_HOME/lib:${LD_LIBRARY_PATH}"
    #export CMAKE_PREFIX_PATH="$OPENBLAS_HOME:${CMAKE_PREFIX_PATH}"
    export C_INCLUDE_DIR="$OPENBLAS_HOME/include"
    export CPLUS_INCLUDE_DIR="$OPENBLAS_HOME/include"

    : ================== Installing Pytorch ==================
    export _GLIBCXX_USE_CXX11_ABI=1
    # git clone --recursive https://github.com/pytorch/pytorch.git -b v${TORCH_VERSION}
    # cd pytorch
    # sed -i '/lintrunner ;/s/$/ and platform_machine != "ppc64le"/' requirements.txt
    # uv pip install -r requirements.txt \
    #    --extra-index-url "$IBM_DEVPI_URL" \
    #    --index-strategy unsafe-best-match \
    #    --no-build-isolation
    # python setup.py develop
    # rm -f dist/torch*+git*whl
    # MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    # PYTORCH_BUILD_VERSION=${TORCH_VERSION} PYTORCH_BUILD_NUMBER=1 uv build --wheel --out-dir ${WHEEL_DIR}

    # cd ${TEMP_BUILD_DIR}

    : ================== Installing Torchvision ==================
    # export TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_FFMPEG=0
    # git clone --recursive https://github.com/pytorch/vision.git -b v${TORCHVISION_VERSION}
    # cd vision
    # uv pip install standard-pkg-resources --no-build-isolation
    # MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    # BUILD_VERSION=${TORCHVISION_VERSION} \
    # uv build --wheel --out-dir ${WHEEL_DIR} --no-build-isolation

    # cd ${TEMP_BUILD_DIR}

    : ================== Installing Torchaudio ==================
    export BUILD_SOX=1 BUILD_KALDI=1 BUILD_RNNT=1 USE_FFMPEG=0 USE_ROCM=0 USE_CUDA=0
    export TORCHAUDIO_TEST_ALLOW_SKIP_IF_NO_FFMPEG=1
    git clone --recursive https://github.com/pytorch/audio.git -b v${TORCHAUDIO_VERSION}
    cd audio
    #patching 
    sed -i '
    s|_CSRC_DIR / "_torchaudio.cpp"|str(_CSRC_DIR / "_torchaudio.cpp")|;
    s|_CSRC_DIR / "utils.cpp"|str(_CSRC_DIR / "utils.cpp")|;
    s|sources=\[_CSRC_DIR / s for s in sources\]|sources=[str(_CSRC_DIR / s) for s in sources]|;
    ' tools/setup_helpers/extension.py
    MAX_JOBS=${MAX_JOBS:-$(nproc)} \
    BUILD_VERSION=${TORCHAUDIO_VERSION} \
    uv build --wheel --out-dir ${WHEEL_DIR} --no-build-isolation

    cd ${REPO_ROOT}
    rm -rf ${TEMP_BUILD_DIR}

#    try_install_from_devpi torchvision
#    try_install_from_devpi torchaudio
}

# TODO(): figure out exact llvmlite version needed by numba
install_llvmlite() {
    curl -L -o ${WHEEL_DIR}/llvmlite-0.47.0-2-cp312-cp312-linux_ppc64le.whl   https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.5-EA1/cpu-ubi9-test/llvmlite-0.47.0-2-cp312-cp312-linux_ppc64le.whl
    # uv pip install "https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.5-EA2/cpu-ubi9-test/llvmlite-0.47.0-2-cp312-cp312-linux_ppc64le.whl"
    # if try_install_from_devpi llvmlite==0.44.0; then
    #    return
    # fi

    # TEMP_BUILD_DIR=$(mktemp -d)
    # cd $TEMP_BUILD_DIR

    # export LLVMLITE_VERSION=${LLVMLITE_VERSION:-0.44.0}

    # TEMP_BUILD_DIR=$(mktemp -d)
    # cd ${TEMP_BUILD_DIR}

    # : ================== Installing Llvmlite ==================
    # git clone --recursive https://github.com/numba/llvmlite.git -b v${LLVMLITE_VERSION}
    # cd llvmlite
    # echo "setuptools<70.0.0" > build_constraints.txt
    # uv build --wheel --out-dir /llvmlitewheel --build-constraint build_constraints.txt

    # : ================= Fix LLvmlite Wheel ====================
    # cd /llvmlitewheel

    # auditwheel repair llvmlite*.whl
    # mv wheelhouse/llvmlite*.whl ${WHEEL_DIR}

    # cd "$REPO_ROOT"
    # rm -rf $TEMP_BUILD_DIR
}


install_opencv() {
    export OPENCV_VERSION=${OPENCV_VERSION:-4.13.0.92}
    echo "========== Installing OpenCV from DevPI =========="
    try_install_from_devpi "opencv-python-headless==${OPENCV_VERSION}"
    echo "OpenCV installed from DevPI"
    # TEMP_BUILD_DIR=$(mktemp -d)
    # cd ${TEMP_BUILD_DIR}
    # export OPENCV_VERSION=92

    # export ENABLE_HEADLESS=1
    # git clone --recursive https://github.com/opencv/opencv-python.git -b ${OPENCV_VERSION} && \
    # cd opencv-python && \
    # if  [[ ${OPENCV_VERSION} == "92" ]]; then sed -i 's/__ARCH_PWR10__/__ARCH_PWR10__)/' opencv/modules/core/include/opencv2/core/vsx_utils.hpp; fi && \
    # sed -i -E -e 's/"setuptools.+",/"setuptools",/g' pyproject.toml && \
    # uv build --wheel --out-dir ${WHEEL_DIR} && \
    # cd "$REPO_ROOT" && \
    # rm -rf $TEMP_BUILD_DIR

}



########################################
# PYARROW
########################################

install_pyarrow() {

TEMP_BUILD_DIR=$(mktemp -d)
cd $TEMP_BUILD_DIR

PYARROW_VERSION=$(curl -s https://api.github.com/repos/apache/arrow/releases/latest | jq -r '.tag_name' | grep -Eo "[0-9.]+")

git clone --depth 1 https://github.com/apache/arrow.git -b apache-arrow-${PYARROW_VERSION}

cd arrow/cpp
mkdir build && cd build

cmake -DCMAKE_BUILD_TYPE=release \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DARROW_PYTHON=ON \
-DARROW_PARQUET=ON \
-DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make install -j ${MAX_JOBS}

cd ../../python
export PYARROW_BUNDLE_ARROW_CPP=1
export PATH=/opt/vllm/bin:$PATH
export ARROW_HOME=/usr/local
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig
export CMAKE_ARGS="-DPython3_EXECUTABLE=/opt/vllm/bin/python"


uv pip install libcst==1.8.6
uv pip install -r requirements-wheel-build.txt \
    --extra-index-url "$IBM_DEVPI_URL" \
    --index-strategy unsafe-best-match \
    --no-build-isolation

pip install "cython>=3.1"

python -m build --wheel --outdir ${WHEEL_DIR} --no-isolation

cd "$REPO_ROOT"
rm -rf $TEMP_BUILD_DIR
}

########################################
# NUMBA
########################################

# install_numba() {

# # TEMP_BUILD_DIR=$(mktemp -d)
# # cd $TEMP_BUILD_DIR

# #NUMBA_VERSION=$(grep -Eo '^numba.+;' $REPO_ROOT/requirements/cpu.txt | grep -Eo '[0-9.]+' | tail -1)
# NUMBA_VERSION=$(grep 'numba' "$REPO_ROOT/requirements/cpu.txt" | \
#     sed -E 's/.*numba *== *([0-9.]+).*/\1/' | \
#     tail -1)
# # git clone --depth 1 https://github.com/numba/numba.git -b ${NUMBA_VERSION}

# # cd numba

# # sed -i '/#include "internal\/pycore_atomic.h"/i\#include "dynamic_annotations.h"' numba/_dispatcher.cpp || true

# # uv build --wheel --out-dir ${WHEEL_DIR} --no-build-isolation

# # cd "$REPO_ROOT"
# # rm -rf $TEMP_BUILD_DIR
# uv pip install numba==0.65.0
# }

install_xgrammar() {
    curl -L -o ${WHEEL_DIR}/xgrammar-0.2.0-2-cp312-cp312-linux_ppc64le.whl   https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.5-EA1/cpu-ubi9-test/xgrammar-0.2.0-2-cp312-cp312-linux_ppc64le.whl
    # uv pip install "https://packages.redhat.com/api/pulp-content/public-rhai/rhoai/3.5-EA2/cpu-ubi9-test/xgrammar-0.2.0-2-cp312-cp312-linux_ppc64le.whl"
    # cd ${REPO_ROOT}

    # echo "========== Installing xgrammar =========="
    # export XGRAMMAR_VERSION="0.1.32"
    # echo "xgrammar version: ${XGRAMMAR_VERSION}"

    # TEMP_BUILD_DIR=$(mktemp -d)
    # cd ${TEMP_BUILD_DIR}

    # export CFLAGS="-fno-lto -mcpu=power9"
    # export CXXFLAGS="-fno-lto -mcpu=power9"
    # export LDFLAGS="-fno-lto"
    # export CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF"

    # echo "========== Cloning xgrammar =========="
    # git clone --recursive https://github.com/mlc-ai/xgrammar -b v${XGRAMMAR_VERSION}

    # cd xgrammar

    # cp cmake/config.cmake .

    # echo "========== Building wheel =========="
    # uv build --wheel --out-dir ${WHEEL_DIR}

    # echo "========== Installing wheel =========="
    # uv pip install ${WHEEL_DIR}/xgrammar*.whl

    # echo "========== Cleanup =========="
    # cd ${REPO_ROOT}
    # rm -rf ${TEMP_BUILD_DIR}

}


########################################
# RUN BUILDS
########################################
install_opencv
install_torch_family
install_llvmlite
# install_pyarrow
# install_numba
install_xgrammar

########################################
# install built wheels
########################################
uv pip install setuptools_scm maturin setuptools-rust ninja scikit-build-core pybind11 nanobind \
    --no-build-isolation

uv pip install ${WHEEL_DIR}/*.whl \
    --extra-index-url "$IBM_DEVPI_URL" \
    --index-strategy unsafe-best-match \
    --no-build-isolation

#sed -i.bak -e 's/.*torch.*//g' pyproject.toml requirements/*.txt
#uv pip install ${WHEEL_DIR}/*.whl || true

########################################
# install remaining deps
########################################

sed -i.bak -e 's/.*torch.*//g' pyproject.toml requirements/*.txt


uv pip install httptools \
    --extra-index-url "$IBM_DEVPI_URL" \
    --index-strategy unsafe-best-match \
    --no-build-isolation || true

# revert back for numba/llvmlite compatibility
uv pip install "setuptools<70" --no-build-isolation

export PKG_CONFIG_PATH=$(find / -type d -name "pkgconfig" 2>/dev/null | tr '\n' ':')

uv pip install -r requirements/common.txt \
               -r requirements/cpu.txt \
               -r requirements/build/cpu.txt --extra-index-url "$IBM_DEVPI_URL" --index-strategy unsafe-best-match
