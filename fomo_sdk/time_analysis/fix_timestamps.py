import argparse
import os
import numpy as np
from rosbags.rosbag2 import Reader
from tqdm import tqdm
from matplotlib import pyplot as plt
from fomo_sdk.common.fomo_mcap_writer import Writer
import fomo_sdk.common.utils as utils
import re
import yaml
import shutil

from typing import cast

from rosbags.interfaces import ConnectionExtRosbag2


def fix_sensors_timestamps(path: str, topics: str, overwrite: bool) -> tuple:
    estop_values = {}
    estop_timestamps = {}
    topics_timestamps = {}
    typestore = utils.get_fomo_typestore()
    input_split = path.split("/")
    output_path = "/".join(input_split[:-1] + ["fixed_timestamps_" + input_split[-1]])

    topics_offsets = {}
    with open(topics) as file:
        yaml_dict = yaml.safe_load(file)
        for topic_regex in yaml_dict.keys():
            topics_offsets[re.compile(topic_regex)] = yaml_dict[topic_regex]

    print(f"Processing: Fixing timestamps in {path}")

    if os.path.exists(output_path) and overwrite:
        shutil.rmtree(output_path)

    with Reader(path) as reader, Writer(output_path, version=8) as writer:
        total_messages = reader.message_count
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
        topic_map = {}
        for connection, timestamp, rawdata in tqdm(
            reader.messages(connections=reader.connections),
            total=total_messages,
            desc="Processing input data",
        ):
            if connection.topic not in topic_map.keys():
                matched = False
                for pattern, values in topics_offsets.items():
                    if pattern.match(connection.topic):
                        topic_map[connection.topic] = values.copy()
                        matched = True
                        break
                if not matched:
                    topic_map[connection.topic] = {}

            values = topic_map[connection.topic]
            offset_value = int(values.get("offset", 0))

            if values.get("ignore", False):
                continue

            outdata = rawdata
            timestamp_out = timestamp + offset_value * 1e9  # int in nanoseconds
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            if connection.topic == "/radar/b_scan_msg":
                if offset_value == 0:
                    timestamp_out = (
                        msg.b_scan_img.header.stamp.sec * 1e9
                        + msg.b_scan_img.header.stamp.nanosec
                    )
                else:
                    msg.b_scan_img.header.stamp.sec += offset_value
                    timestamp_out = (
                        msg.b_scan_img.header.stamp.sec * 1e9
                        + msg.b_scan_img.header.stamp.nanosec
                    )
                    outdata: memoryview | bytes = typestore.serialize_cdr(
                        msg, connection.msgtype
                    )
            elif head := getattr(msg, "header", None):
                if (
                    offset_value == 0
                ):  # no offset value, just use the header timestamp as the bag timestamp
                    timestamp_out = head.stamp.sec * 1e9 + head.stamp.nanosec
                else:
                    msg.header.stamp.sec += offset_value
                    timestamp_out = (
                        msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
                    )
                outdata: memoryview | bytes = typestore.serialize_cdr(
                    msg, connection.msgtype
                )

            writer.write(conn_map[connection.id], int(timestamp_out), outdata)

            if connection.topic in [
                "/teleop/emergency_stop",
                "/warthog/platform/emergency_stop",
            ]:
                if connection.topic not in estop_values.keys():
                    estop_values[connection.topic] = []
                    estop_timestamps[connection.topic] = []
                estop_values[connection.topic].append(msg.data)
                estop_timestamps[connection.topic].append(timestamp_out)
            if connection.topic not in topics_timestamps.keys():
                topics_timestamps[connection.topic] = []
            topics_timestamps[connection.topic].append(timestamp_out)

    for topic in estop_timestamps.keys():
        estop_timestamps[topic] = np.array(estop_timestamps[topic])
        estop_values[topic] = np.array(estop_values[topic])
    for topic in topics_timestamps.keys():
        topics_timestamps[topic] = np.array(topics_timestamps[topic])
    return (topics_timestamps, estop_timestamps, estop_values)


def main(rosbag_path: str, topics: str, overwrite: bool):
    topics_timestamps, estop_timestamps, estop_values = fix_sensors_timestamps(
        rosbag_path, topics, overwrite
    )

    try:
        # Plotting
        fig1, ax1 = plt.subplots(figsize=(15, 10))
        experiment_start = 0
        experiment_start = min(
            [np.min(timestamps) for timestamps in topics_timestamps.values()]
        )
        for i, (topic, timestamps) in enumerate(sorted(topics_timestamps.items())):
            ax1.plot(
                timestamps - experiment_start,
                i * np.ones(len(timestamps)),
                "-.",
                label=topic,
            )
        # Set text labels for y axis
        ax1.set_yticks(np.arange(len(topics_timestamps)))
        ax1.set_yticklabels(sorted(topics_timestamps.keys()))
        ax1.set_xlabel("Timestamps")
        ax1.set_ylabel("Topics")
        ax1.set_title("Topic Timestamps Plot")
        ax1.axis("tight")  # Adjust axis to fit data tightly
        fig1.subplots_adjust(left=0.2)
        fig1.savefig("/Users/mbo/Desktop/topics_timestamps.png")
        plt.show()
        plt.close(fig1)  # Close the figure
    except Exception as e:
        print(e)

    try:
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        for topic in estop_timestamps.keys():
            ax2.plot(estop_timestamps[topic], estop_values[topic], "-.", label=topic)
        ax2.legend()
        ax2.set_xlabel("Timestamps")
        ax2.set_ylabel("E-Stop Values")
        ax2.set_title("E-Stop Timestamps Plot")
        ax2.axis("tight")  # Adjust axis to fit data tightly
        ax2.ticklabel_format(useOffset=False)
        ax2.ticklabel_format(style="plain")
        plt.xticks(rotation=30)
        plt.show()
        fig2.savefig("/Users/mbo/Desktop/estop_timestamps.png")
        plt.close(fig2)  # Close the figure
    except Exception as e:
        print(e)

    print("Plot saved to topics_timestamps.png and estop_timestamps.png")

    try:
        first_estop_min = None
        last_estop_max = None
        for topic in estop_timestamps.keys():
            # Get first timestamp where estop_values == 0
            if first_estop_min is None:
                first_estop_min = estop_timestamps[topic][
                    np.where(estop_values[topic] == 0)[0][0]
                ]
                last_estop_max = estop_timestamps[topic][
                    np.where(estop_values[topic] == 1)[0][-1]
                ]
            else:
                first_estop_min = min(
                    estop_timestamps[topic][np.where(estop_values[topic] == 0)[0][0]],
                    first_estop_min,
                )
                last_estop_max = max(
                    estop_timestamps[topic][np.where(estop_values[topic] == 1)[0][-1]],
                    last_estop_max,
                )

        np.set_printoptions(suppress=True)
        print(f"Start: {first_estop_min / 1e9}, end: {last_estop_max / 1e9}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TODO.")
    parser.add_argument(
        "-p", "--rosbag_path", type=str, help="Path pointing to a ROS 2 bag file."
    )
    parser.add_argument(
        "-t",
        "--topics_offsets",
        type=str,
        help="A path to a .yaml file containing list of topics with timestamp correction to process.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help="Overwrite existing output bag.",
        action="store_true",
    )
    args = parser.parse_args()

    main(args.rosbag_path, args.topics_offsets, args.overwrite)
