"""
Exports a static IMU db9 rosbag file for calibration purposes, that contains IMU data from the interval <bag_start, param_start>.
"""

import argparse
import os
from rosbags.rosbag2 import Reader
from tqdm import tqdm
import fomo_sdk.common.utils as utils
from fomo_sdk.common.fomo_mcap_writer import Writer
import fomo_sdk.imu.imu_utils as iutils
import shutil

from typing import cast

from rosbags.interfaces import ConnectionExtRosbag2


def export(input_paths: str, output_path: str, start: float, overwrite: bool):
    typestore = utils.get_fomo_typestore()

    if os.path.exists(output_path) and overwrite:
        shutil.rmtree(output_path)

    writer = Writer(output_path, version=8)
    writer.open()
    for path in input_paths:
        path = path[0]
        start_bag = start
        with Reader(path) as reader:
            conn_map = {}
            total_messages = reader.message_count

            if start_bag is None:
                start_bag = reader.start_time / 1e9

            print(f"Saving IMU messages for calibration to {output_path}")

            # convert start and end time to ns to match rosbag timestamps
            start_bag = int(start_bag * 1e9)

            for conn in reader.connections:
                ext = cast(ConnectionExtRosbag2, conn.ext)
                if conn.msgtype == iutils.Imu.__msgtype__:
                    conn_map[conn.id] = writer.add_connection(
                        conn.topic,
                        conn.msgtype,
                        serialization_format=ext.serialization_format,
                        offered_qos_profiles=ext.offered_qos_profiles,
                        typestore=typestore,
                    )

            for connection, timestamp, rawdata in tqdm(
                reader.messages(connections=reader.connections),
                total=total_messages,
                desc=f"Processing {path}",
            ):
                iutils.write_imu_data(
                    writer,
                    conn_map,
                    rawdata,
                    connection,
                    timestamp,
                    start_bag,
                )

    writer.close()


def main(input: str, output_path: str, start: float, overwrite: bool):
    export(input, output_path, start, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export IMU topics between input bag start and the start parameter to the output bagfile."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path pointing to a ROS 2 bag file.",
        action="append",
        nargs="+",
    )
    parser.add_argument("-o", "--output", required=True, type=str, help="Output path.")
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing output bag.",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=float,
        help="Start of the rosbag, in seconds.",
        default=None,
    )
    args = parser.parse_args()

    main(args.input, args.output, args.start, args.overwrite)
