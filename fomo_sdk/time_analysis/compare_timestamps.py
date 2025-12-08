from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

from rosbags.rosbag2 import Reader

# label, rate = ("/lslidar128/points", 10)
# label, rate = ("/radar/b_scan_msg", 4)
# label, rate = ("/basler/driver/image_raw", 10)
# label, rate = ("/zed_node/left_raw/image_raw_color", 5)
label, rate = ("/mti30/data_raw", 200)
# label, rate = ("/vn100/data_raw", 800)

title = label.split("/")[1]

bagpath_ptp = Path("data/fomo_test_mti30")
# bagpath_ptp = Path('data/fomo_test_vn100_startup_time_800hz')
bagpath_no_ptp = Path("data/fomo_test_vn100_ros_time_800hz")


def get_timestamps_from_bag(bagpath, topic):
    timestamps = []
    # Create reader instance and open for reading.
    with Reader(bagpath) as reader:
        # Topic and msgtype information is available on .connections list.
        # for connection in reader.connections:
        #     print(connection.topic, connection.msgtype)

        # Iterate over messages.
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic:
                timestamps.append(timestamp)
            # if connection.topic == '/imu_raw/Imu':
            #     msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            #     print(msg.header.frame_id)

        # The .messages() method accepts connection filters.
        # connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
        # for connection, timestamp, rawdata in reader.messages(connections=connections):
        #     msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        #     print(msg.header.frame_id)
    return timestamps


dict = {
    "Device time": get_timestamps_from_bag(bagpath_ptp, label),
    "ROS time": get_timestamps_from_bag(bagpath_no_ptp, label),
}

dict_sorted = {}
for label, timestamps in dict.items():
    dict_sorted[label] = np.array(sorted(timestamps))
    dict_sorted[label] = np.diff(dict_sorted[label])
    dict_sorted[label] = dict_sorted[label] / 1e6  # Convert to ms
    dict_sorted[label] -= 1000 / rate

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig.suptitle(title)
ax1.boxplot(dict_sorted.values(), labels=dict_sorted.keys())
ax1.set_ylabel("Time delay between messages [ms]")
# ax1.legend()


labels = {}
for label, timestamps in dict.items():
    dict_sorted[label] /= 1000 / rate
    labels[label] = f"{label}\n({rate} Hz)"
ax2.boxplot(dict_sorted.values(), labels=labels.values())
ax2.set_ylabel("Time delay/message period")
ax2.legend()

plt.show()
