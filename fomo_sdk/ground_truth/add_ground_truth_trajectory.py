"""
Adds ground truth trajectory data from a .tum format file to a rosbag.
The script reads a .tum trajectory file and converts it to ROS 2 messages:
- geometry_msgs/msg/PoseStamped messages for each pose at the specified timestamp
- nav_msgs/msg/Path message containing all poses (published once with latching QoS)

Additionally copies specific topics from the input rosbag to preserve mapping data.

Example usage:
    python fomo_sdk/add_ground_truth_trajectory.py -i input.mcap -t trajectory.tum -o output.mcap
"""

import argparse
import os
import shutil
from typing import cast
from tqdm import tqdm

from rosbags.rosbag2 import Reader
from rosbags.interfaces import ConnectionExtRosbag2
from rosbags.typesys.stores.latest import (
    builtin_interfaces__msg__Time as Time,
    std_msgs__msg__Header as Header,
    geometry_msgs__msg__Point as Point,
    geometry_msgs__msg__Quaternion as Quaternion,
    geometry_msgs__msg__Pose as Pose,
    geometry_msgs__msg__PoseStamped as PoseStamped,
    geometry_msgs__msg__Transform as Transform,
    geometry_msgs__msg__TransformStamped as TransformStamped,
    geometry_msgs__msg__Vector3 as Vector3,
    nav_msgs__msg__Path as Path,
    tf2_msgs__msg__TFMessage as TFMessage,
)

import fomo_sdk.common.utils as utils
from fomo_sdk.common.fomo_mcap_writer import Writer


# Topics to copy from input rosbag
TOPICS_TO_COPY = ["/map", "/icp_odom", "/tf", "/tf_static", "/robot_description"]


def read_tum_file(tum_file_path: str) -> list:
    """
    Read a .tum format trajectory file.

    Args:
        tum_file_path: Path to the .tum file

    Returns:
        List of tuples containing (timestamp, x, y, z, qx, qy, qz, qw)
    """
    trajectory_data = []

    with open(tum_file_path, "r") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                parts = line.split()
                if len(parts) != 8:
                    print(
                        f"Warning: Line {line_num} has {len(parts)} values, expected 8. Skipping."
                    )
                    continue

                timestamp = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                    float(parts[7]),
                )

                trajectory_data.append((timestamp, x, y, z, qx, qy, qz, qw))

            except ValueError as e:
                print(f"Warning: Error parsing line {line_num}: {e}. Skipping.")
                continue

    print(f"Loaded {len(trajectory_data)} poses from {tum_file_path}")
    return trajectory_data


def create_pose_stamped_message(
    timestamp: float,
    x: float,
    y: float,
    z: float,
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    frame_id: str = "world",
) -> PoseStamped:
    """
    Create a PoseStamped message from trajectory data.

    Args:
        timestamp: Timestamp in seconds (with nanosecond precision)
        x, y, z: Position coordinates
        qx, qy, qz, qw: Quaternion orientation
        frame_id: Frame ID for the pose

    Returns:
        PoseStamped message
    """
    # Convert timestamp to ROS Time
    # Input timestamp is in nanoseconds, convert to seconds and nanoseconds
    timestamp_ns = int(timestamp)
    sec = timestamp_ns // 1000000000
    nanosec = timestamp_ns % 1000000000

    header = Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)

    position = Point(x=x, y=y, z=z)
    orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    pose = Pose(position=position, orientation=orientation)

    return PoseStamped(header=header, pose=pose)


def create_path_message(trajectory_data: list, frame_id: str = "world") -> Path:
    """
    Create a Path message containing all poses from the trajectory.

    Args:
        trajectory_data: List of trajectory tuples
        frame_id: Frame ID for the path

    Returns:
        Path message
    """
    if not trajectory_data:
        return Path()

    # Use the first timestamp for the path header
    first_timestamp = trajectory_data[0][0]
    timestamp_ns = int(first_timestamp)
    sec = timestamp_ns // 1000000000
    nanosec = timestamp_ns % 1000000000

    header = Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)

    poses = []
    for timestamp, x, y, z, qx, qy, qz, qw in trajectory_data:
        pose_stamped = create_pose_stamped_message(
            timestamp, x, y, z, qx, qy, qz, qw, frame_id
        )
        poses.append(pose_stamped)

    return Path(header=header, poses=poses)


def create_map_to_world_transform(
    first_pose_data: tuple, frame_id: str = "world"
) -> TransformStamped:
    """
    Create a TransformStamped message from map to world frame based on the first pose.

    Args:
        first_pose_data: First trajectory tuple (timestamp, x, y, z, qx, qy, qz, qw)
        frame_id: Target frame ID for the transform

    Returns:
        TransformStamped message
    """
    timestamp, x, y, z, qx, qy, qz, qw = first_pose_data

    # Convert timestamp to ROS Time
    timestamp_ns = int(timestamp)
    sec = timestamp_ns // 1000000000
    nanosec = timestamp_ns % 1000000000

    header = Header(stamp=Time(sec=sec, nanosec=nanosec), frame_id=frame_id)

    translation = Vector3(x=x, y=y, z=z)
    rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    transform = Transform(translation=translation, rotation=rotation)

    return TransformStamped(header=header, child_frame_id="map", transform=transform)


def add_ground_truth_trajectory(
    input_bag_path: str,
    tum_file_path: str,
    output_bag_path: str,
    overwrite: bool,
    frame_id: str = "world",
    duration: float = None,
):
    """
    Add ground truth trajectory from .tum file to rosbag.

    Args:
        input_bag_path: Path to input rosbag
        tum_file_path: Path to .tum trajectory file
        output_bag_path: Path to output rosbag
        overwrite: Whether to overwrite existing output
        frame_id: Frame ID for the trajectory messages
        duration: Duration in seconds to process (None for all data)
    """
    if not os.path.exists(input_bag_path):
        raise FileNotFoundError(f"Input bag file not found: {input_bag_path}")

    if not os.path.exists(tum_file_path):
        raise FileNotFoundError(f"TUM trajectory file not found: {tum_file_path}")

    if os.path.exists(output_bag_path) and overwrite:
        shutil.rmtree(output_bag_path)
    elif os.path.exists(output_bag_path):
        raise FileExistsError(
            f"Output path exists: {output_bag_path}. Use --overwrite to replace."
        )

    # Read trajectory data
    trajectory_data = read_tum_file(tum_file_path)
    if not trajectory_data:
        raise ValueError("No valid trajectory data found in TUM file")

    # Filter trajectory data by duration if specified
    if duration is not None:
        start_time = trajectory_data[0][0] / 1e9  # Convert to seconds
        end_time = start_time + duration
        trajectory_data = [
            pose for pose in trajectory_data if pose[0] / 1e9 <= end_time
        ]
        print(
            f"Filtered trajectory to {len(trajectory_data)} poses within {duration}s duration"
        )

    typestore = utils.get_fomo_typestore()

    writer = Writer(output_bag_path, version=8)
    writer.set_compression(
        Writer.CompressionMode.MESSAGE, Writer.CompressionFormat.ZSTD
    )
    writer.open()

    try:
        with Reader(input_bag_path) as reader:
            conn_map = {}

            # Copy specified topics from input bag
            for connection in reader.connections:
                if connection.topic in TOPICS_TO_COPY:
                    ext = cast(ConnectionExtRosbag2, connection.ext)
                    conn_map[connection.id] = writer.add_connection(
                        connection.topic,
                        connection.msgtype,
                        serialization_format=ext.serialization_format,
                        offered_qos_profiles=ext.offered_qos_profiles,
                        typestore=typestore,
                    )

            # Create connections for ground truth trajectory
            pose_connection = writer.add_connection(
                "/ground_truth/pose",
                PoseStamped.__msgtype__,
                typestore=typestore,
            )

            # Create path connection with default QoS
            path_connection = writer.add_connection(
                "/ground_truth/path",
                Path.__msgtype__,
                typestore=typestore,
            )

            # Copy messages from input bag and add map-to-world transform to tf_static
            total_input_messages = reader.message_count
            map_to_world_written = False
            first_timestamp_ns = int(trajectory_data[0][0])

            # Calculate end timestamp for duration filtering
            end_timestamp_ns = None
            if duration is not None:
                end_timestamp_ns = first_timestamp_ns + int(duration * 1e9)

            for connection, timestamp, rawdata in tqdm(
                reader.messages(connections=reader.connections),
                total=total_input_messages,
                desc="Processing original rosbag",
                unit=" msgs",
            ):
                if connection.id in conn_map:
                    # Skip messages outside duration if specified
                    if end_timestamp_ns is not None and timestamp > end_timestamp_ns:
                        continue
                    writer.write(conn_map[connection.id], timestamp, rawdata)

                    # Add map-to-world transform to tf_static topic (write once)
                    if connection.topic == "/tf_static" and not map_to_world_written:
                        map_to_world_transform = create_map_to_world_transform(
                            trajectory_data[0]
                        )
                        tf_static_msg = TFMessage(transforms=[map_to_world_transform])
                        writer.write(
                            conn_map[connection.id],
                            first_timestamp_ns,
                            typestore.serialize_cdr(
                                tf_static_msg, TFMessage.__msgtype__
                            ),
                        )
                        map_to_world_written = True

            # Write individual pose messages and incremental path messages
            accumulated_poses = []
            last_path_time = 0

            for i, (timestamp, x, y, z, qx, qy, qz, qw) in enumerate(
                tqdm(
                    trajectory_data,
                    desc="Writing ground truth trajectory",
                    unit=" poses",
                )
            ):
                pose_msg = create_pose_stamped_message(
                    timestamp, x, y, z, qx, qy, qz, qw, frame_id
                )
                timestamp_ns = int(timestamp)
                writer.write(
                    pose_connection,
                    timestamp_ns,
                    typestore.serialize_cdr(pose_msg, PoseStamped.__msgtype__),
                )

                # Add pose to accumulated poses
                accumulated_poses.append(pose_msg)

                # Write incremental path every 2 seconds
                current_time = timestamp_ns / 1e9
                if (
                    current_time - last_path_time >= 2.0
                    or i == len(trajectory_data) - 1
                ):
                    # Create path message with accumulated poses
                    path_header = Header(
                        stamp=Time(
                            sec=timestamp_ns // 1000000000,
                            nanosec=timestamp_ns % 1000000000,
                        ),
                        frame_id=frame_id,
                    )
                    incremental_path_msg = Path(
                        header=path_header, poses=accumulated_poses.copy()
                    )
                    writer.write(
                        path_connection,
                        timestamp_ns,
                        typestore.serialize_cdr(incremental_path_msg, Path.__msgtype__),
                    )
                    last_path_time = current_time

    finally:
        writer.close()


def main(
    input_bag: str,
    tum_file: str,
    output_bag: str,
    overwrite: bool,
    frame_id: str,
    duration: float,
):
    """Main function to process the trajectory and rosbag."""
    add_ground_truth_trajectory(
        input_bag, tum_file, output_bag, overwrite, frame_id, duration
    )
    utils.notify_user(
        os.path.basename(__file__), f"Ground truth trajectory added to {output_bag}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add ground truth trajectory from .tum file to rosbag. "
        "Copies mapping-related topics from input bag and adds ground truth pose data."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input ROS 2 bag file (mcap format)",
    )
    parser.add_argument(
        "-t", "--tum_file", type=str, required=True, help="Path to .tum trajectory file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output ROS 2 bag file (mcap format)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output bag"
    )
    parser.add_argument(
        "--frame_id",
        type=str,
        default="world",
        help="Frame ID for the ground truth trajectory (default: world)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds to process from the beginning (default: process all data)",
    )

    args = parser.parse_args()
    main(
        args.input,
        args.tum_file,
        args.output,
        args.overwrite,
        args.frame_id,
        args.duration,
    )
