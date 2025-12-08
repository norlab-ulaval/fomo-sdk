import argparse
import datetime
import os
import shutil

import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys.stores.latest import builtin_interfaces__msg__Time as Time
from rosbags.typesys.stores.latest import sensor_msgs__msg__NavSatFix as NavSatFix
from rosbags.typesys.stores.latest import sensor_msgs__msg__NavSatStatus as NavSatStatus
from rosbags.typesys.stores.latest import std_msgs__msg__Header as Header
from tqdm import tqdm

from fomo_sdk.common import utils
from fomo_sdk.common.fomo_mcap_writer import Writer
from fomo_sdk.ground_truth.utils import open_pos_file


# Define helper functions
def to_ros_time(date_str, time_str):
    dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f")
    timestamp = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return timestamp


def main(
    input_file,
    rosbag_path,
    start,
    end,
    rosbag_timestamp_path,
    overwrite,
    topic_namespace,
):
    if (start is not None or end is not None) and rosbag_timestamp_path is not None:
        raise Exception(
            "Cannot specify both start/end and a rosbag path to automatically determine a timestamp. Please choose one."
        )
    if rosbag_timestamp_path is not None:
        with Reader(rosbag_timestamp_path) as reader:
            start = reader.start_time / 1e9
            end = reader.end_time / 1e9
    # Check for overwrite condition
    if os.path.exists(rosbag_path) and not overwrite:
        print(f"File {rosbag_path} already exists. Use --overwrite to overwrite.")
        return

    if os.path.exists(rosbag_path) and overwrite:
        shutil.rmtree(rosbag_path)

    typestore = utils.get_fomo_typestore()

    FoMoSatelliteStatusIdx = typestore.types["fomo_msgs/msg/NavsatSatelliteStatus"]

    # Read file and convert data
    with Writer(rosbag_path, version=8) as writer:
        for emlid_position in ["front", "left", "right"]:
            gnss_connection_fix = writer.add_connection(
                f"/{topic_namespace}/{emlid_position}",
                NavSatFix.__msgtype__,
                typestore=typestore,
            )
            satellite_status = writer.add_connection(
                f"/{topic_namespace}/{emlid_position}/status",
                FoMoSatelliteStatusIdx.__msgtype__,
                typestore=typestore,
            )
            df = open_pos_file(os.path.join(input_file, emlid_position + ".pos"))
            # tqdm progress bar for each position
            for row in tqdm(
                list(df.itertuples()), desc=f"Writing /emlid/{emlid_position}"
            ):
                # Construct NavSatFix message
                timestamp = row.timestamp / 1e9
                time = Time(
                    sec=int(np.floor(timestamp)),
                    nanosec=int((timestamp - np.floor(timestamp)) * 1e9),
                )
                header = Header(time, f"emlid_m2_{emlid_position}")
                status = NavSatStatus(status=int(2), service=NavSatStatus.SERVICE_GPS)
                fomo_satellite_status_msg = FoMoSatelliteStatusIdx(
                    header=header,
                    status=row.Q,
                    num_sats=row.ns,
                    age=row.age,
                    ratio=row.ratio,
                )
                navsat_msg = NavSatFix(
                    header,
                    status,
                    row.latitude,
                    row.longitude,
                    row.altitude,
                    np.array(
                        [
                            float(row.cov_xx),
                            float(row.cov_xy),
                            float(row.cov_xz),
                            float(row.cov_yx),
                            float(row.cov_yy),
                            float(row.cov_yz),
                            float(row.cov_zx),
                            float(row.cov_zy),
                            float(row.cov_zz),
                        ]
                    ),
                    NavSatFix.COVARIANCE_TYPE_KNOWN,
                )

                # Write messages to bag
                writer.write(
                    gnss_connection_fix,
                    timestamp=int(timestamp * 1e9),
                    data=typestore.serialize_cdr(navsat_msg, NavSatFix.__msgtype__),
                )
                writer.write(
                    satellite_status,
                    timestamp=int(timestamp * 1e9),
                    data=typestore.serialize_cdr(
                        fomo_satellite_status_msg, FoMoSatelliteStatusIdx.__msgtype__
                    ),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GNSS data to a ROS 2 bag file."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path containing a file called back.pos, front.pos and middle.pos.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output ROS 2 bag file path.",
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing output bag.",
        action="store_true",
    )
    parser.add_argument(
        "--rosbag_timestamp",
        type=str,
        help="Rosbag used to determine start and end timestamps.",
    )
    parser.add_argument(
        "-s", "--start", type=float, help="Start of the rosbag, in seconds."
    )
    parser.add_argument(
        "-e", "--end", type=float, help="End of the rosbag, in seconds."
    )
    parser.add_argument(
        "--topic_namespace",
        type=str,
        default="emlid",
        help="Namespace to prepend to the topic name.",
    )
    args = parser.parse_args()

    main(
        args.input,
        args.output,
        args.start,
        args.end,
        args.rosbag_timestamp,
        args.overwrite,
        args.topic_namespace,
    )
