FROM mcr.microsoft.com/devcontainers/go:0-1-bullseye
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates cmake curl unzip nano wget g++ git zsh

# Install Libtorch - CPU
RUN wget https://raw.githubusercontent.com/sugarme/gotch/master/setup-libtorch.sh
RUN chmod +x setup-libtorch.sh
RUN sed -i 's/sudo//g' setup-libtorch.sh
ENV CUDA_VER=cpu
RUN bash setup-libtorch.sh

ENV GOTCH_LIBTORCH="/usr/local/lib/libtorch"
ENV LIBRARY_PATH="$LIBRARY_PATH:$GOTCH_LIBTORCH/lib"
ENV CPATH="$CPATH:$GOTCH_LIBTORCH/lib:$GOTCH_LIBTORCH/include:$GOTCH_LIBTORCH/include/torch/csrc/api/include"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$GOTCH_LIBTORCH/lib"

# Setup gotch
RUN wget https://raw.githubusercontent.com/sugarme/gotch/master/setup-gotch.sh
RUN chmod +x setup-gotch.sh
RUN sed -i 's/sudo//g' setup-gotch.sh
ENV CUDA_VER=cpu
ENV GOTCH_VER=v0.7.0
RUN bash setup-gotch.sh

RUN go clean
RUN go clean -cache

ENTRYPOINT zsh