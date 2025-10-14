# VERA GPU Micro Applications

VERA GPU Micro Applications is a C++ project containing GPU-accelerated micro-applications for nuclear reactor physics simulations. The project contains 4 micro-applications: MOC (Method of Characteristics), CMFD (Coarse Mesh Finite Difference), Conduction (heat conduction), and Subchannel (thermal-hydraulic analysis).

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

**⚠️ IMPORTANT: Use the CI container `bradenpecora/vera-gpu:0.0.1no-cuda` for all testing. This is the ONLY environment where the full build and test suite will work correctly. Manual dependency installation will fail due to missing PETSc-Kokkos integration.**

## Working Effectively

- **CRITICAL BUILD REQUIREMENTS**: This project requires custom-built dependencies and CANNOT be built with standard Ubuntu packages alone. The Ubuntu PETSc packages lack required Kokkos integration headers.

- **RECOMMENDED: Use CI Container for Testing**
  - The CI uses the Docker container `bradenpecora/vera-gpu:0.0.1no-cuda` which has all dependencies pre-installed via Spack
  - **To use the CI container for testing:**
    - `docker pull bradenpecora/vera-gpu:0.0.1no-cuda` -- pulls the CI container image
    - `docker run -it --rm -v /home/runner/work/vera_gpu_micro_app/vera_gpu_micro_app:/workspace bradenpecora/vera-gpu:0.0.1no-cuda` -- runs container with repo mounted
    - Inside container: `source /opt/spack-environment/activate.sh` -- activates Spack environment with all dependencies
    - Inside container: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx` -- configures project (~6 seconds)
    - Inside container: `cmake --build build --parallel 4` -- builds project (~90 seconds)
    - Inside container: `cd build && ctest -j4 --output-on-failure -L CONTINUOUS` -- runs CI tests (~71 seconds, 26 tests)
  - **Container details:**
    - Base: Ubuntu 22.04
    - Dependencies installed via Spack: PETSc (with Kokkos integration), Kokkos, HighFive, HDF5, OpenMPI
    - Environment activation script: `/opt/spack-environment/activate.sh` (must be sourced before building)
    - All dependencies in: `/opt/views/view/` (symlinked to `/opt/view`)
  - **This is the PREFERRED method for testing as it matches the CI environment exactly**

- **Alternative: Bootstrap, build, and test the repository manually (EXPECTED TO FAIL):**
  - `sudo apt-get update` -- takes ~10 seconds
  - `sudo apt-get install -y build-essential cmake git curl wget gfortran libhdf5-dev libopenmpi-dev openmpi-bin` -- takes ~60 seconds. NEVER CANCEL.
  - `export HDF5_ROOT=/usr/bin`
  - **Install Kokkos (custom build required):**
    - `cd /tmp && git clone --branch develop https://github.com/kokkos/kokkos.git` -- takes ~3 seconds
    - `sudo cmake -B kokkos/build -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_INSTALL=ON -DCMAKE_INSTALL_PREFIX=/opt/kokkos kokkos` -- takes ~1 second
    - `sudo cmake --build kokkos/build --parallel 6` -- takes ~15 seconds. NEVER CANCEL. Set timeout to 120+ seconds.
    - `sudo cmake --install kokkos/build` -- takes <1 second
    - `export KOKKOS_ROOT=/opt/kokkos`
    - `rm -rf kokkos`
  - **Install PETSc (NETWORK BLOCKED - will fail):**
    - `export PETSC_DIR=/opt/petsc && export PETSC_ARCH=arch-linux-opt`
    - `sudo git clone -b release https://gitlab.com/petsc/petsc.git $PETSC_DIR` -- **WILL FAIL: gitlab.com is blocked**
    - **Alternative: Use Ubuntu packages (incomplete but allows partial build):**
      - `sudo apt-get install -y libpetsc-real-dev` -- takes ~18 minutes. NEVER CANCEL. Set timeout to 1800+ seconds.
      - `export PETSC_DIR=/usr/lib/petscdir/petsc3.19 && export PETSC_ARCH=x86_64-linux-gnu-real`
  - **Install HighFive:**
    - `cd /tmp && git clone --recursive --branch v2.9.0 https://github.com/BlueBrain/HighFive.git` -- takes ~4 seconds
    - `sudo cmake -B HighFive/build HighFive -DCMAKE_INSTALL_PREFIX=/opt/highfive -DHIGHFIVE_BUILD_DOCS=OFF -DHIGHFIVE_BUILD_TESTS=OFF -DHIGHFIVE_USE_BOOST=OFF` -- takes <1 second
    - `sudo cmake --build HighFive/build --parallel 6` -- takes ~48 seconds. NEVER CANCEL. Set timeout to 300+ seconds.
    - `sudo cmake --install HighFive/build` -- takes <1 second
    - `export HIGHFIVE_ROOT=/opt/highfive`
    - `rm -rf HighFive`
  - **Configure and build the project:**
    - `mkdir -p build`
    - `cmake -B build -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx` -- takes ~4 seconds
    - `cmake --build build -j4` -- **WILL FAIL: missing petscmat_kokkos.hpp**. This is expected with Ubuntu PETSc packages.

- **Run tests (will fail due to build failure):**
  - `cd build && ctest -j4 --output-on-failure -L CONTINUOUS` -- takes <1 second. All tests will show "Not Run" due to missing executables.

## Validation

### Using CI Container (Recommended)
- **All tests pass in CI container**: 26 tests run successfully in ~71 seconds
- **Test Labels in CI**:
  - `CONTINUOUS`: Tests run in CI pipeline (26 tests, ~264 seconds total)
  - `BASIC`: Quick fundamental tests (17 tests, ~44 seconds total)
  - `HEAVY`: Resource-intensive tests (26 tests, ~264 seconds total)
  - Micro-app specific: `MOC` (14 tests), `CMFD` (6 tests), `Conduction` (3 tests), `Subchannel` (1 test), `Utils` (2 tests)
- **Expected container setup time**: Pull container (~1-2 minutes first time), build (~90 seconds), test (~71 seconds)
- **Container provides**: Full Spack-built environment with PETSc+Kokkos integration

### Manual Build (Not Recommended)
- **NEVER CANCEL LONG OPERATIONS**: 
  - Dependency installation takes up to 20 minutes total
  - PETSc Ubuntu package installation specifically takes ~18 minutes
  - Always set timeout to at least 1800 seconds for PETSc installation
- **Expected Build Failure**: The project CANNOT be fully built in this environment due to missing PETSc-Kokkos integration. This is a known limitation.
- **Testing Limitation**: Tests cannot run because executables fail to build due to missing headers.
- **Network Limitations**: gitlab.com is blocked, preventing custom PETSc installation from source.

## Common Tasks

The following are key directory structures and important files:

### Repository Root
```
.
├── .github/
├── .gitignore
├── CMakeLists.txt          # Main build configuration
├── README.md               # Comprehensive setup documentation
├── LICENSE
├── cmake/                  # Custom CMake find modules
├── utils/                  # Shared utilities library
├── cmfd/                   # CMFD micro-application
├── moc/                    # MOC micro-application
├── conduction/            # Conduction micro-application
└── subchannel/            # Subchannel micro-application
```

### Micro-Application Structure
Each micro-app (cmfd/, moc/, conduction/, subchannel/) contains:
```
├── CMakeLists.txt         # Build configuration
├── src/                   # Source code
├── tests/                 # Unit tests
└── data/                  # Test data (HDF5 files)
```

### Key Build Files
- `CMakeLists.txt` - Main project configuration, sets up dependencies
- `cmake/FindPETSc.cmake` - Custom PETSc finder (expects custom PETSc build)
- `cmake/FindHighFive.cmake` - HighFive finder with HDF5 fallback
- `cmake/TestFunctions.cmake` - Helper functions for test creation

### CI Workflow
The CI workflow (`.github/workflows/testing.yml`) runs on every push and uses:
- **Container**: `bradenpecora/vera-gpu:0.0.1no-cuda`
- **Steps**:
  1. Checkout with LFS support (`actions/checkout@v4` with `lfs: true`)
  2. Configure: `source /opt/spack-environment/activate.sh && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx`
  3. Build: `source /opt/spack-environment/activate.sh && cmake --build build --parallel 4`
  4. Test: `cd build && ctest -j4 --output-on-failure -L CONTINUOUS`
- **Build time**: ~90 seconds
- **Test time**: ~71 seconds (26 tests)
- **Success criteria**: All CONTINUOUS labeled tests pass

### Test Labels
- `BASIC`: Quick fundamental tests (17 tests in CI)
- `CONTINUOUS`: Tests run in CI pipeline (26 tests)
- `HEAVY`: Resource-intensive tests (26 tests in CI)
- Micro-app specific: `MOC` (14 tests), `CMFD` (6 tests), `Conduction` (3 tests), `Subchannel` (1 test), `Utils` (2 tests)

## Testing Strategy

### Primary Method: CI Container
Use the CI container (`bradenpecora/vera-gpu:0.0.1no-cuda`) for all testing to match the CI environment exactly:
```bash
docker pull bradenpecora/vera-gpu:0.0.1no-cuda
docker run -it --rm -v $(pwd):/workspace bradenpecora/vera-gpu:0.0.1no-cuda
source /opt/spack-environment/activate.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx
cmake --build build --parallel 4
cd build && ctest -j4 --output-on-failure -L CONTINUOUS
```

### Running Specific Tests in Container
```bash
# Run only BASIC tests (faster, ~44 seconds)
cd build && ctest -j4 --output-on-failure -L BASIC

# Run specific micro-app tests
cd build && ctest -j4 --output-on-failure -L MOC
cd build && ctest -j4 --output-on-failure -L CMFD

# Run a single test by name
cd build && ctest --output-on-failure -R test_exp_table

# List all available tests
cd build && ctest -N
```

### Debugging Build Issues in Container
```bash
# View CMake configuration details
cd build && cmake -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_VERBOSE_MAKEFILE=ON

# Build with verbose output
cmake --build build --parallel 4 --verbose

# Check which dependencies are found
cd build && cmake .. -LA | grep -E "KOKKOS|PETSC|HIGHFIVE|HDF5"
```

### Alternative Methods (When Container Cannot Be Used)
If the container is not accessible:
1. **For code analysis**: Focus on source files in `*/src/` directories
2. **For testing**: Review test files in `*/tests/` directories  
3. **For documentation**: Check README.md for complete setup instructions
4. **For CI**: Review `.github/workflows/testing.yml` (uses Docker with prebuilt dependencies)

## Dependencies Summary

### CI Container Dependencies (via Spack)
The `bradenpecora/vera-gpu:0.0.1no-cuda` container includes:
- **PETSc**: Built with flags `cflags='-O3 -fopenmp' cppflags='-O3 -fopenmp' +kokkos`
  - Includes Kokkos integration (critical requirement)
  - Linked with: `^hdf5+hl+mpi ^hwloc ^kokkos+openmp+serial ^openmpi`
- **HighFive**: Built with `+mpi ^hdf5+hl+mpi ^hwloc ^openmpi`
- **Kokkos**: Configured with `+openmp+serial` (no CUDA in this container)
- **HDF5**: Built with `+hl+mpi` (hierarchical data format with MPI support)
- **OpenMPI**: MPI implementation
- **CMake**: Build system (3.16+)
- **GCC**: C++ compiler with C++17 support

### Manual Build Dependencies
- **C++ Compiler**: GCC with C++17 support
- **MPI**: OpenMPI (mpicc, mpicxx compilers)
- **CMake**: 3.16+ 
- **Kokkos**: GPU/parallel computing framework (custom build required)
- **PETSc**: Numerical solvers (custom build with Kokkos support required)
- **HighFive**: HDF5 C++ wrapper (custom build required)
- **HDF5**: Data format library (Ubuntu package sufficient)
- **GoogleTest**: Unit testing (fetched automatically)

## Environment Variables Required
```bash
export HDF5_ROOT=/usr/bin
export KOKKOS_ROOT=/opt/kokkos
export PETSC_DIR=/opt/petsc                    # Custom build path
export PETSC_ARCH=arch-linux-opt              # Custom build architecture
export HIGHFIVE_ROOT=/opt/highfive
```

**Note**: With Ubuntu packages, use:
```bash
export PETSC_DIR=/usr/lib/petscdir/petsc3.19
export PETSC_ARCH=x86_64-linux-gnu-real
```

This setup allows configuration but build will fail due to missing Kokkos integration headers.