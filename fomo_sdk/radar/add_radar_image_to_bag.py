import argparse
import os
from rosbags.rosbag2 import Reader
from tqdm import tqdm
import fomo_sdk.common.utils as utils
from fomo_sdk.common.fomo_mcap_writer import Writer
import yaml
import shutil
import fomo_sdk.radar.utils as rutils


from typing import cast
from rosbags.interfaces import ConnectionExtRosbag2


def main(path: str, overwrite: bool):
    typestore = utils.get_fomo_typestore()
    input_split = path.split("/")
    output_path = "/".join(input_split[:-1] + ["radar_" + input_split[-1]])

    print(f"Processing: Filtering {path} based on e-stop signal")
    with open(os.path.join(path, "metadata.yaml")) as file:
        yaml_dict = yaml.safe_load(file)
        total_messages = yaml_dict["rosbag2_bagfile_information"][
            "message_count"
        ]  # Get total count for tqdm

    if os.path.exists(output_path) and overwrite:
        shutil.rmtree(output_path)

    with Reader(path) as reader, Writer(output_path, version=8) as writer:
        conn_map = {}
        for conn in reader.connections:
            ext = cast(ConnectionExtRosbag2, conn.ext)
            conn_map[conn.id] = writer.add_connection(
                conn.topic,
                conn.msgtype,
                serialization_format=ext.serialization_format,
                offered_qos_profiles=ext.offered_qos_profiles,
                typestore=typestore,
            )
        radar_image_connection = writer.add_connection(
            "/radar/image",
            rutils.Image.__msgtype__,
            typestore=typestore,
        )
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=reader.connections),
            total=total_messages,
            desc="Processing input data",
        ):
            if connection.topic == "/radar/b_scan_msg":
                rutils.write_radar_image(
                    writer,
                    radar_image_connection,
                    connection,
                    rawdata,
                    timestamp,
                    typestore,
                )
            writer.write(conn_map[connection.id], timestamp, rawdata)
    print(f"Filtered rosbag saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO.")
    parser.add_argument(
        "-p", "--rosbag_path", type=str, help="Path pointing to a ROS 2 bag file."
    )
    parser.add_argument("-o", "--overwrite", action="store_true")
    args = parser.parse_args()

    main(args.rosbag_path, args.overwrite)
