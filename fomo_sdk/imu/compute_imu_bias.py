"""
Exports a static IMU db9 rosbag file for calibration purposes, that contains IMU data from the interval <bag_start, param_start>.
"""

import argparse
import json
from rosbags.rosbag2 import Reader
from tqdm import tqdm
import fomo_sdk.common.utils as utils
import fomo_sdk.imu.imu_utils as iutils


class ImuData:
    def __init__(self, topic: str):
        self.topic = topic
        self.data = {
            "ang_x": [],
            "ang_y": [],
            "ang_z": [],
        }

    def add_data(self, msg):
        self.data["ang_x"].append(msg.angular_velocity.x)
        self.data["ang_y"].append(msg.angular_velocity.y)
        self.data["ang_z"].append(msg.angular_velocity.z)

    def compute_bias(self):
        bias_data = {}
        for key, value in self.data.items():
            bias = sum(value) / len(value)
            print(f"{key} has bias {bias}")
            # Map the keys to match the desired JSON format
            if key == "ang_x":
                bias_data["x"] = bias
            elif key == "ang_y":
                bias_data["y"] = bias
            elif key == "ang_z":
                bias_data["z"] = bias
        return bias_data


def main(input_paths: str, start: float, end: float, output_file: str = None):
    typestore = utils.get_fomo_typestore()
    imu_data_dict = {}

    for path in input_paths:
        path = path[0]
        with Reader(path) as reader:
            if start is None:
                start = reader.start_time / 1e9
            if end is None:
                end = reader.end_time / 1e9

            for connection, timestamp, rawdata in tqdm(
                reader.messages(
                    connections=reader.connections,
                    start=int(start * 1e9),
                    stop=int(end * 1e9),
                )
            ):
                if connection.msgtype == iutils.Imu.__msgtype__:
                    if connection.topic not in imu_data_dict:
                        imu_data_dict[connection.topic] = ImuData(connection.topic)
                    imu_data_dict[connection.topic].add_data(
                        typestore.deserialize_cdr(rawdata, connection.msgtype)
                    )

    # Create the JSON output structure
    json_output = {}

    for topic, imu_data in imu_data_dict.items():
        print(f"Topic: {imu_data.topic}")
        bias_data = imu_data.compute_bias()

        # Extract the device name from the topic
        # Handle formats like '/vectornav/data_raw', '/xsens/imu', etc.
        topic_parts = topic.strip("/").split("/")
        if len(topic_parts) >= 2:
            # Use the first part (device name) instead of the last part
            device_name = topic_parts[0]
        elif len(topic_parts) == 1:
            device_name = topic_parts[0]
        else:
            device_name = topic.strip("/")

        # Remap device names
        device_mapping = {"mti30": "xsens", "vn100": "vectornav"}
        device_name = device_mapping.get(device_name, device_name)

        json_output[device_name] = {"angular_velocity": bias_data}

    # Export to JSON file or print to stdout
    if output_file:
        with open(output_file, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nJSON output saved to: {output_file}")
    else:
        print("\nJSON Output:")
        print(json.dumps(json_output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export IMU topics between input bag start and the start parameter to JSON format."
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
    parser.add_argument(
        "-s",
        "--start",
        type=float,
        help="Start of the rosbag, in seconds.",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--end",
        type=float,
        help="End of the rosbag, in seconds.",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output JSON file path. If not specified, prints to stdout.",
        default=None,
    )

    args = parser.parse_args()
    main(args.input, args.start, args.end, args.output)
