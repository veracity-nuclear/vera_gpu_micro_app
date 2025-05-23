# vera_gpu_micro_app

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependency Installation](#installing-dependencies)
3. [Package Installation](#installation-instructions)
4. [Developer Tools](#developer-tools)


## Project Overview
Contains micro-applications for prototyping of VERA GPU capabilities

Four micro-apps are defined in this repo, each with its own source
and test data:
1. MOC
2. CMFD
3. Conduction
4. Subchannel

Each of these focuses on a different computationally intensive portion
of the VERA solve.  Test data produced from VERA can be used to test
the accuracy and performance of the implementation.


## Dependency Installation
### Build essentials, CMake, OpenMP and HDF5 from apt package manager
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git curl wget \
    gfortran libhdf5-dev libopenmpi-dev openmpi-bin
```

### Kokkos
1. Clone the Kokkos repository
```
git clone --branch develop https://github.com/kokkos/kokkos.git
```

2. Configure the build with CMake (may need to sudo for access to /opt directory)
```
cmake -B kokkos/build \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkos_ENABLE_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX=/opt/kokkos \
  kokkos
```

3. Build Kokkos (any number of processors can be used)
```
cmake --build kokkos/build --parallel 6
```

4. Install Kokkos
```
cmake --install kokkos/build
```

5. Cleanup (optional)
```
rm -rf kokkos
```

### PETSc
1. Set environment variables
```
export PETSC_DIR=/opt/petsc
export PETSC_ARCH=arch-linux-opt
```

2. Clone the PETSc repository (may need to sudo for access to /opt directory)
```
git clone -b release https://gitlab.com/petsc/petsc.git $PETSC_DIR
cd $PETSC_DIR
```

3. Configure the build (this may take >30 min to complete)
```
./configure \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=0 \
  --with-debugging=0 \
  --with-rtlib=libc \
  --download-mpich \
  --download-hdf5 \
  --download-hwloc \
  --download-f2cblaslapack \
  --COPTFLAGS="-O3 -march=native" \
  --CXXOPTFLAGS="-O3 -march=native"
```

4. Build PETSc
```
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
```

5. Confirm the installation
```
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
```

### HighFive
1. Clone the HighFive repository
```
git clone --recursive --branch v2.9.0 https://github.com/BlueBrain/HighFive.git
```

2. Configure the build with CMake (may need to sudo for access to /opt directory)
```
cmake -B HighFive/build HighFive \
  -DCMAKE_INSTALL_PREFIX=/opt/highfive \
  -DHIGHFIVE_BUILD_DOCS=OFF \
  -DHIGHFIVE_BUILD_TESTS=OFF \
  -DHIGHFIVE_USE_BOOST=OFF
```

3. Build HighFive (any number of processors can be used)
```
cmake --build HighFive/build --parallel 6
```

4. Install HighFive
```
cmake --install HighFive/build
```

5. Cleanup (optional)
```
rm -rf HighFive
```


## Package Configuration and Build
From `~/vera_gpu_micro_app` execute the following commands to configure and build the micro-apps.
```
mkdir -p build
cmake -B build
cmake --build build -j4
```


## Developer Tools
