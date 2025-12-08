# Fomo Rust SDK

## Dependencies
This crate depends on OpenCV. Detailed guide can be found [here](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md).To install OpenCV, follow the instructions for your operating system:

### MacOS

```shell
brew install opencv
```

then, you will need to manually specify the path to the OpenCV library:

```shell
export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select --print-path)/usr/lib/"
export SDKROOT=$(xcrun --show-sdk-path)
```

### Linux

```shell
sudo apt install libopencv-dev clang libclang-dev
```

## Installation
cargo install --path .

## Usage
```shell
mcap_to_ijrr --help
Usage: mcap_to_ijrr [OPTIONS] --input <INPUT> --output <OUTPUT> --drivetrain <DRIVETRAIN>

Options:
  -i, --input <INPUT>
  -o, --output <OUTPUT>
  -s, --sensors <SENSORS>        Sensor types to process. Valid options: all, audio, basler, leishen, navtech, robosense, zedx_left, zedx_right
  -d, --drivetrain <DRIVETRAIN>  Sensor types to process. Valid options: wheels, tracks
  -h, --help                     Print help
  -V, --version                  Print version
```

```shell
Usage: ijrr_to_mcap [OPTIONS] --input <INPUT> --output <OUTPUT>

Options:
  -i, --input <INPUT>
  -o, --output <OUTPUT>
  -s, --sensors <SENSORS>  Sensor types to process. Valid options: all, audio, basler, leishen, navtech, robosense, zedx_left, zedx_right
  -h, --help               Print help
  -V, --version            Print version
```

## Docker
```shell
docker build -t fomo-rust .
docker run -it fomo-rust
```
