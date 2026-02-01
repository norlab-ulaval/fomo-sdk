# --- Stage 1: Builder ---
FROM rust:1.89-bullseye AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libopencv-dev \
    pkg-config \
    libclang-dev \
    llvm \
    llvm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/fomo-sdk
COPY fomo_sdk/data ./fomo_sdk/data

WORKDIR /usr/src/fomo-sdk/fomo_rust
COPY fomo_rust/Cargo.toml fomo_rust/Cargo.lock ./
COPY fomo_rust/src ./src
COPY fomo_rust/benches ./benches
COPY fomo_rust/tests ./tests

ENV RUSTFLAGS="-C target-cpu=generic"
RUN cargo install --path .

# --- Stage 2: Runtime ---
FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y \
    libopencv-video-dev \
    libopencv-highgui-dev \
    libopencv-imgproc-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The binary remains in the same relative target folder
COPY --from=builder /usr/src/fomo-sdk/fomo_rust/target/release/ijrr_to_mcap /usr/local/bin/ijrr_to_mcap

ENTRYPOINT ["ijrr_to_mcap"]
CMD ["--help"]
