# docker build -t your-dockerhub-username/vera-gpu:tag -f vera_gpu.dockerfile .
# docker push your-dockerhub-username/vera-gpu:tag
# docker tag your-dockerhub-username/vera-gpu:tag your-dockerhub-username/vera-gpu:latest
# docker push your-dockerhub-username/vera-gpu:latest
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git curl wget python3 \
    gfortran libhdf5-openmpi-dev openmpi-bin libopenmpi-dev \
    libomp-dev cuda-toolkit-12-4

ENV CC=mpicc
ENV CXX=mpicxx

# --- Kokkos (with CUDA + OpenMP) ---
RUN git clone --branch develop https://github.com/kokkos/kokkos.git && \
    cmake -B kokkos/build -S kokkos \
        -DKokkos_ENABLE_OPENMP=ON \
        -DKokkos_ENABLE_CUDA=ON \
        -DKokkos_ARCH_AMPERE86=ON \
        -DKokkos_ENABLE_CUDA_LAMBDA=ON \
        -DKokkos_ENABLE_INSTALL=ON \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DKokkos_ENABLE_LIBDL=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/kokkos \
        -DCMAKE_CXX_COMPILER=g++ \
        -DKokkos_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CXX_STANDARD=17 && \
    cmake --build kokkos/build --parallel && \
    cmake --install kokkos/build && \
    rm -rf kokkos

ENV Kokkos_DIR=/opt/kokkos/lib/cmake/Kokkos

# --- HighFive ---
RUN git clone --recursive --branch v2.9.0 https://github.com/BlueBrain/HighFive.git && \
    cmake -B HighFive/build HighFive \
        -DCMAKE_INSTALL_PREFIX=/opt/highfive \
        -DHIGHFIVE_BUILD_DOCS=OFF \
        -DHIGHFIVE_BUILD_TESTS=OFF \
        -DHIGHFIVE_USE_BOOST=OFF && \
    cmake --build HighFive/build --parallel && \
    cmake --install HighFive/build && \
    rm -rf HighFive

ENV HighFive_DIR=/opt/highfive/share/HighFive/CMake/

# --- PETSc ---
ENV PETSC_DIR=/opt/petsc \
    PETSC_ARCH=arch-linux-opt \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN git clone -b release https://gitlab.com/petsc/petsc.git ${PETSC_DIR} && \
    cd ${PETSC_DIR} && \
    ./configure \
        --with-cc=mpicc \
        --with-cxx=mpicxx \
        --with-fc=0 \
        --with-debugging=0 \
        --with-mpi=1 \
        --with-hdf5=1 \
        --with-hdf5-include=/usr/include/hdf5/openmpi \
        --with-hdf5-lib="-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -lhdf5_openmpi -lhdf5_openmpi_hl" \
        --download-hwloc \
        --download-f2cblaslapack \
        --COPTFLAGS="-O3 -march=native" \
        --CXXOPTFLAGS="-O3 -march=native" && \
    make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all

ENV PATH=${PETSC_DIR}/${PETSC_ARCH}/bin:$PATH
ENV LD_LIBRARY_PATH=${PETSC_DIR}/${PETSC_ARCH}/lib:$LD_LIBRARY_PATH

# --- Git LFS ---
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# --- OpenMP Environment Variables for Best Performance for OpenMP 4.0 or better ---
ENV OMP_PROC_BIND=spread \
    OMP_PLACES=threads

# Default workdir for projects
WORKDIR /workspace