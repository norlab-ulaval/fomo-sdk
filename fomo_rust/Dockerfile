FROM rust:1.89-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    libopencv-dev \
    pkg-config \
    libclang-dev \
    llvm \
    llvm-dev


RUN git clone https://github.com/norlab-ulaval/fomo-sdk.git /usr/src/fomo-sdk
WORKDIR /usr/src/fomo-sdk/fomo_rust
COPY src/ /usr/src/fomo-sdk/fomo_rust/src
COPY benches/ /usr/src/fomo-sdk/fomo_rust/benches
COPY tests/ /usr/src/fomo-sdk/fomo_rust/tests
COPY Cargo.toml /usr/src/fomo-sdk/fomo_rust/Cargo.toml
COPY Cargo.lock /usr/src/fomo-sdk/fomo_rust/Cargo.lock

ENV RUSTFLAGS="-C target-cpu=generic"
RUN cargo install --path .

CMD ["ijrr_to_mcap", "--help"]
