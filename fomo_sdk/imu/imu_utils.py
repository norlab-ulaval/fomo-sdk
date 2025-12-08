from rosbags.typesys.stores.latest import sensor_msgs__msg__Imu as Imu


def write_imu_data(writer, conn_map, rawdata, connection, timestamp: int, start: int):
    """
    Writes IMU messages before the start timestamp (in seconds)
    """
    try:
        if connection.msgtype == Imu.__msgtype__ and timestamp < start:
            writer.write(conn_map[connection.id], timestamp, rawdata)
    except OverflowError as e:
        print(f"Start: {start}, Timestamp: {timestamp}")
        raise e
