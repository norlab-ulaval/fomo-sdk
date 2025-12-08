import argparse
import os
from tqdm import tqdm
import cv2
import fomo_sdk.radar.utils as rutils


def polar_to_cartesian(input: str, output: str):
    if not os.path.exists(output):
        os.makedirs(output)
    for file in tqdm(os.listdir(input)):
        if file.endswith(".png"):
            input_path = os.path.join(input, file)
            print(f"Processing {input_path}...")
            (_, azimuths, fft_data) = rutils.load(input_path)
            print(fft_data.shape, azimuths.shape)
            cartesian = rutils.polar_to_cartesian(
                fft_data,
                azimuths,
                cart_resolution=0.2384,
                cart_pixel_width=1024,
            )
            cv2.imwrite(os.path.join(output, file), cartesian)
            print(f"Processed {input_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path pointing to a folder containing polar radar data.",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output path for the processed data."
    )
    args = parser.parse_args()

    polar_to_cartesian(args.input, args.output)
