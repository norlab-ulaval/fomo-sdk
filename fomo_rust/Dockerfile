FROM rust:1.85.1-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libopencv-dev \
    pkg-config \
    libclang-dev \
    llvm \
    llvm-dev


WORKDIR /usr/src/fomo_rust
COPY src/ /usr/src/fomo_rust/src
COPY benches/ /usr/src/fomo_rust/benches
COPY tests/ /usr/src/fomo_rust/tests
COPY Cargo.toml /usr/src/fomo_rust/Cargo.toml
COPY Cargo.lock /usr/src/fomo_rust/Cargo.lock

RUN rustup default stable
RUN rustup update
RUN cargo install --path .

CMD ["ijrr_to_mcap", "--help"]
