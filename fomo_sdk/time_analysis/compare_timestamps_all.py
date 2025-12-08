from matplotlib import pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import polars as pl
import yaml
import numpy as np
from tqdm import tqdm

from rosbags.rosbag2 import Reader
import fomo_sdk.common.utils as utils

showfliers = False

colors = {"PTP": "blue", "ROS": "red", "DEVICE": "green", "PTP->ROS": "orange"}


def get_timestamps_from_bag(bagpath, topics):
    typestore = utils.get_fomo_typestore()
    topic_timestamps = {topic: [] for topic in topics}
    with Reader(bagpath) as reader:
        i = 0
        for connection, timestamp, rawdata in tqdm(
            reader.messages(), total=reader.message_count, desc="Processing input data"
        ):
            i += 1
            if i > reader.message_count:
                break
            if connection.topic in topics:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                if connection.topic == "/radar/b_scan_msg":
                    timestamp = (
                        msg.b_scan_img.header.stamp.sec
                        + msg.b_scan_img.header.stamp.nanosec / 1e9
                    )
                else:
                    timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                topic_timestamps[connection.topic].append(timestamp)

    # convert to sorted numpy array
    for topic, timestamps in topic_timestamps.items():
        topic_timestamps[topic] = np.array(sorted(timestamps))
    return topic_timestamps


def color_background(ax, dict_sorted, df):
    # Loop over each boxplot to add a background color
    for i, label in enumerate(dict_sorted.keys()):
        # Calculate x position of the boxplot
        x_position = i + 1  # Boxplot x positions are 1-based
        width = 0.5  # Width of the background rectangle
        height = ax.get_ylim()[1]  # Height from the bottom to the top of the plot

        timesource = df.filter(pl.col("label") == label)["timesource"].to_list()[0]
        color = colors[timesource]

        # Add a rectangle as background
        rect = patches.Rectangle(
            (x_position - width / 2, ax.get_ylim()[0]),
            width,
            2 * height,
            color=color,
            alpha=0.3,
        )
        ax.add_patch(rect)


def main():
    with open("sensors_rate_analysis.yaml", "r") as file:
        bagfile_labels_rates = yaml.safe_load(file)

    rows = []
    for label, values in bagfile_labels_rates.items():
        row = [label] + list(values.values())
        rows.append(row)

    columns = ["label", "timesource", "bagfile", "topic", "rate"]
    df = pl.DataFrame(rows, schema=columns, orient="row")
    df = df.with_columns(pl.col("timesource").replace(colors).alias("color"))
    df_bag_names = df.unique(subset=["bagfile"])["bagfile"]

    timestamps_dict = {}
    for bagfile in df_bag_names:
        topics = pl.Series(
            df.filter(pl.col("bagfile") == bagfile).select(["topic"])
        ).to_list()
        bagpath = Path(bagfile)
        if not bagpath.exists():
            print(f"Bagfile {bagpath} does not exist.")
        else:
            print(f"Processing {bagpath}...")
            timestamps_dict = timestamps_dict | get_timestamps_from_bag(bagpath, topics)
    dict_sorted = {}
    labels = {}
    for topic, timestamps in timestamps_dict.items():
        # get row from df
        row = df.filter(pl.col("topic") == topic)
        label = row["label"].to_list()[0]
        rate = row["rate"].to_list()[0]
        dict_sorted[label] = np.abs(np.diff(timestamps_dict[topic]))
        dict_sorted[label] = dict_sorted[label] / 1e6  # Convert to ms
        # dict_sorted[label] -= 1000 / rate
        if "zedx" not in label:
            labels[label] = f"{label.split('_')[0]}\n({rate} Hz)"
        else:
            labels[label] = f"{label}\n({rate} Hz)"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, tight_layout=True, figsize=(19, 10)
    )
    ax1.boxplot(dict_sorted.values(), showfliers=showfliers, labels=labels.values())
    ax1.set_ylabel("Difference from expected message delay [ms]")
    color_background(ax1, dict_sorted, df)

    # convert to the difference from expected message delay [ms]
    for label in dict_sorted.keys():
        row = df.filter(pl.col("label") == label)
        rate = row["rate"].to_list()[0]
        dict_sorted[label] /= 1000 / rate
    ax2.boxplot(dict_sorted.values(), labels=labels.values(), showfliers=showfliers)
    ax2.set_ylabel("Time delay/message period")

    legend_patches = [
        patches.Patch(
            color=list(colors.values())[i], label=list(colors.keys())[i], alpha=0.3
        )
        for i in range(len(colors))
    ]
    ax2.legend(handles=legend_patches, loc="upper right")
    color_background(ax2, dict_sorted, df)

    plt.show()
    # if showfliers:
    #     fig.savefig("./time_analysis_fliers.png", dpi=300)
    # else:
    #     fig.savefig("./time_analysis_no_fliers.png", dpi=300)


if __name__ == "__main__":
    main()
