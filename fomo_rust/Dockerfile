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

WORKDIR /usr/src/fomo-rust

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY benches ./benches
COPY tests ./tests

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
COPY --from=builder /usr/src/fomo-rust/target/release/ijrr_to_mcap /usr/local/bin/ijrr_to_mcap

ENTRYPOINT ["ijrr_to_mcap"]
CMD ["--help"]
