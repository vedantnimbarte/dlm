# dlm with the CUDA kernels, as a container.
#
#   docker build -t dlm .
#   docker run --gpus all -p 8000:8000 -v /path/to/models:/models dlm \
#       serve --model-path /models/Qwen2.5-0.5B-Instruct --host 0.0.0.0
#
# The binary links the CUDA runtime statically (`cuda-static`), so the image
# needs only the NVIDIA driver, which `--gpus all` brings in from the host.
# Without a GPU it runs on the CPU. `--host 0.0.0.0` makes the port reachable
# from outside the container; set `--api-key` before exposing it further.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
# GPU generations to compile for: Turing to Blackwell (sm_75 to sm_120).
ARG DLM_CUDA_ARCHS=75,80,86,89,90,120
ENV DLM_CUDA_ARCHS=$DLM_CUDA_ARCHS
# Where build.rs finds cudart_static.
ENV CUDA_PATH=/usr/local/cuda
WORKDIR /src
COPY . .
RUN cargo build --release --features cuda-static

FROM ubuntu:24.04
COPY --from=build /src/target/release/dlm /usr/local/bin/dlm
EXPOSE 8000
ENTRYPOINT ["dlm"]
CMD ["--help"]
