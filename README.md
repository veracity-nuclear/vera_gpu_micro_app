# vera_gpu_micro_app

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependency Installation](#dependency-installation)
3. [Package Configuration and Build](#package-configuration-and-build)
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

1. Build packages
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git curl wget \
    gfortran libhdf5-dev libopenmpi-dev openmpi-bin
```

2. Export environment variables
You should export the HDF5 environment variable:
```
export HDF5_ROOT=/usr/bin
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

6. Export environment variables
```
export KOKKOS_ROOT=/opt/kokkos
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

6. Export environment variables
```
export HIGHFIVE_ROOT=/opt/highfive
```


## Package Configuration and Build
Before building, you should `export` all the environment variables defined in the previous sections.

From `~/vera_gpu_micro_app` execute the following commands to configure and build the micro-apps.
```
mkdir -p build
cmake -B build -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build -j4
```


## Developer Tools

### Testing the micro-apps
```
cd build
ctest -j4 --output-on-failure
```

### Running package executables
Execute these commands from the `~/vera_gpu_micro_app/build` directory to run micro-app executables. These can be renamed in
`~/vera_gpu_micro_apps/<micro-app>/CMakeLists.txt` as well.

```
# MOC
./moc/moc_exec

# CMFD
./cmfd/cmfd_exec

# Conduction
./conduction/conduction_exec

# Subchannel
./subchannel/subchannel_exec
```
