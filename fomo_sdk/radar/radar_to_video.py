import argparse
import os
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
from tqdm import tqdm
import fomo_sdk.common.utils as utils
import fomo_sdk.radar.utils as rutils
from rosbags.image import message_to_cvimage


def get_radar_scan_images_and_timestamps(path) -> tuple:
    typestore = utils.get_fomo_typestore()

    radar_times = []
    polar_images = []
    radar_images = []
    print("Processing: Getting image_timestamp and radar image")

    with AnyReader([Path(path)]) as reader:
        connections = [x for x in reader.connections if x.topic == "/radar/b_scan_msg"]
        total_messages = sum(
            1 for _ in reader.messages(connections=connections)
        )  # Get total count for tqdm
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=connections),
            total=total_messages,
            desc="Processing radar data",
        ):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            radar_time = (
                msg.b_scan_img.header.stamp.sec
                + msg.b_scan_img.header.stamp.nanosec / 1e9
            )
            radar_time = round(radar_time, 3)
            radar_times.append(radar_time)

            polar_img = message_to_cvimage(msg.b_scan_img)
            polar_images.append(polar_img)

            azimuths = (
                msg.encoder_values / 5595 * 2 * np.pi
            )  # these are the end bin number for LAVAL radar
            radar_image = rutils.polar_to_cartesian(
                polar_img,
                azimuths,
                cart_resolution=0.2384,
                cart_pixel_width=1024,
            )
            radar_images.append(radar_image)

    return radar_times, polar_images, radar_images


def main(input_path: str, output_path: str):
    radar_times, polar_imgs, radar_imgs = get_radar_scan_images_and_timestamps(
        input_path
    )

    print(
        f"Radar times: {len(radar_times)} frames, first: {radar_times[0]}, last: {radar_times[-1]}"
    )
    print(f"Polar images shape: {np.array(polar_imgs).shape}")
    print(f"Radar images shape: {np.array(radar_imgs).shape}")

    rutils.write_video(
        polar_imgs,
        os.path.join(output_path, "radar_pol.mp4"),
        (2000, 400),
    )
    rutils.write_video(
        radar_imgs,
        os.path.join(output_path, "radar_cart.mp4"),
        radar_imgs[0].shape,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exports polar and cartesian videos from a given rosbag."
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Path pointing to a ROS 2 bag file."
    )
    parser.add_argument("-o", "--output", type=str, help="Output path.")
    args = parser.parse_args()

    main(args.input, args.output)
