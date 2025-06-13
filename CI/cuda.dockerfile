# This file is kind of a work-in-progress. The Dockerfile is currently too big to be used in CI.

# docker build -t your-dockerhub-username/vera-gpu:tag -f Dockerfile .
# docker push your-dockerhub-username/vera-gpu:tag
# docker tag your-dockerhub-username/vera-gpu:tag your-dockerhub-username/vera-gpu:latest
# docker push your-dockerhub-username/vera-gpu:latest

# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-jammy:develop AS builder


# What we want to install and how we want to install it
# is specified in a manifest file (spack.yaml)
RUN mkdir -p /opt/spack-environment && \
set -o noclobber \
&&  (echo spack: \
&&   echo '  # add package specs to the `specs` list' \
&&   echo '  specs:' \
&&   echo '  - petsc cflags='"'"'-O3 -fopenmp'"'"' cppflags='"'"'-O3 -fopenmp'"'"' +cuda+kokkos ^hdf5+hl+mpi ^hwloc+cuda ^kokkos+cuda+openmp+serial+wrapper cuda_arch=86 ^openmpi+cuda' \
&&   echo '  - highfive+mpi ^hdf5+hl+mpi ^hwloc+cuda ^openmpi+cuda' \
&&   echo '  view: /opt/views/view' \
&&   echo '  concretizer:' \
&&   echo '    unify: true' \
&&   echo '  config:' \
&&   echo '    install_tree: /opt/software') > /opt/spack-environment/spack.yaml

# Install the software, remove unnecessary deps
RUN cd /opt/spack-environment && spack env activate . && spack install --fail-fast && spack gc -y

# Strip binaries and create minimal runtime package
RUN find -L /opt/views/view/* -type f -exec readlink -f '{}' \; | \
    xargs file -i | \
    grep 'charset=binary' | \
    grep 'x-executable\|x-archive\|x-sharedlib' | \
    awk -F: '{print $1}' | xargs -I {} sh -c 'strip "{}" 2>/dev/null || true' && \
    cd /opt/spack-environment && \
    spack env activate --sh -d . > activate.sh

# Final minimal runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.4/compat:${LD_LIBRARY_PATH}"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # libgomp1 \
    build-essential \
    curl \
    cmake \
    cuda-toolkit-12-4 \
    ca-certificates && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /opt/spack-environment /opt/spack-environment
COPY --from=builder /opt/software /opt/software

# paths.view is a symlink, so copy the parent to avoid dereferencing and duplicating it
COPY --from=builder /opt/views /opt/views
RUN find /opt/views/view/bin/../targets/x86_64-linux/lib -name "*.a" -exec ranlib {} \;

RUN { \
      echo '#!/bin/sh' \
      && echo '. /opt/spack-environment/activate.sh' \
      && echo 'exec "$@"'; \
    } > /entrypoint.sh \
&& chmod a+x /entrypoint.sh \
&& ln -s /opt/views/view /opt/view

WORKDIR /workspace
ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "/bin/bash" ]