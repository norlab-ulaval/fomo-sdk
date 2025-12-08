import argparse
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import fomo_sdk.common.utils as utils
import re
import os
import yaml


def get_sensors_timestamps(input_path: str, topics: str):
    typestore = utils.get_fomo_typestore()

    try:
        if os.path.exists(topics):
            with open(topics) as file:
                topics_patterns = [re.compile(line.strip()) for line in file]
        else:
            topics_patterns = [re.compile(topics)]
    except Exception as e:
        print(f"Error reading topics file: {e}")
        exit(1)

    topics_timestamps = {}
    for path in input_path:
        path = path[0]
        with open(os.path.join(path, "metadata.yaml")) as file:
            yaml_dict = yaml.safe_load(file)
            total_messages = yaml_dict["rosbag2_bagfile_information"][
                "message_count"
            ]  # Get total count for tqdm
        print(f"Processing: Getting timestamps from {path}")

        with AnyReader([Path(path)]) as reader:
            connections = [
                x
                for x in reader.connections
                if any(pattern.match(x.topic) for pattern in topics_patterns)
            ]
            for connection, timestamp, rawdata in tqdm(
                reader.messages(connections=connections),
                total=total_messages,
                desc="Processing input data",
            ):
                try:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                except (AssertionError, UnicodeDecodeError):
                    print(f"Error deserializing message of '{connection.topic}'")
                    continue
                try:
                    if connection.topic == "/radar/b_scan_msg":
                        timestamp_to_show = (
                            msg.b_scan_img.header.stamp.sec
                            + msg.b_scan_img.header.stamp.nanosec / 1e9
                        )
                    else:
                        timestamp_to_show = (
                            msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                        )

                except AttributeError:
                    timestamp_to_show = timestamp / 1e9
                if connection.topic not in topics_timestamps.keys():
                    topics_timestamps[connection.topic] = []
                topics_timestamps[connection.topic].append(timestamp_to_show)

    for topic in topics_timestamps.keys():
        topics_timestamps[topic] = np.array(topics_timestamps[topic])
    return topics_timestamps


def main(rosbag_path: str, topics: str):
    topics_timestamps = get_sensors_timestamps(rosbag_path, topics)
    experiment_start = 0
    experiment_start = min(
        [np.min(timestamps) for timestamps in topics_timestamps.values()]
    )

    for i, (topic, timestamps) in enumerate(sorted(topics_timestamps.items())):
        plt.plot(
            timestamps - experiment_start,
            i * np.ones(len(timestamps)),
            "o",
            label=topic,
        )
    # Set text labels for y axis
    plt.yticks(
        np.linspace(0, len(topics_timestamps) - 1, len(topics_timestamps)),
        sorted(topics_timestamps.keys()),
    )
    plt.subplots_adjust(left=0.2)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO.")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input path pointing to a ROS 2 bag file.",
        action="append",
        nargs="+",
    )
    parser.add_argument(
        "-t",
        "--topics",
        type=str,
        help="Regex expression OR a path to a .txt file containing list of topics to process.",
    )
    args = parser.parse_args()

    main(args.input, args.topics)
