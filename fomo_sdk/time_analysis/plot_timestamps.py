from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

from rosbags.rosbag2 import Reader
# from rosbags.typesys import Stores, get_typestore

with open("topics.txt") as file:
    topics_rates = {line.strip(): [] for line in file}
    topics = {t_r.split(",")[0].strip(): [] for t_r in topics_rates}
    rates = {
        t_r.split(",")[0].strip(): int(t_r.split(",")[1].strip())
        for t_r in topics_rates
    }

# Create a DataFrame with an empty column for each line in the file
# df = pl.DataFrame({topic: [] for topic in topics})

bagpath = Path("data/fomo_test_basler_ptp")

# Create reader instance and open for reading.
with Reader(bagpath) as reader:
    # Topic and msgtype information is available on .connections list.
    # for connection in reader.connections:
    #     print(connection.topic, connection.msgtype)

    # Iterate over messages.
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic in topics.keys():
            topics[connection.topic].append(timestamp)
        # if connection.topic == '/imu_raw/Imu':
        #     msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        #     print(msg.header.frame_id)

    # The .messages() method accepts connection filters.
    # connections = [x for x in reader.connections if x.topic == '/imu_raw/Imu']
    # for connection, timestamp, rawdata in reader.messages(connections=connections):
    #     msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
    #     print(msg.header.frame_id)

topics_sorted = {}
for topic, timestamps in topics.items():
    print(timestamps)
    topics_sorted[topic] = np.array(sorted(timestamps))
    topics_sorted[topic] = np.diff(topics_sorted[topic])
    topics_sorted[topic] = topics_sorted[topic] / 1e6  # Convert to ms
    topics_sorted[topic] -= 1000 / rates[topic]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.boxplot(topics_sorted.values(), labels=topics_sorted.keys())
ax1.set_ylabel("Time delay between messages [ms]")
# ax1.legend()


labels = {}
for topic, timestamps in topics.items():
    topics_sorted[topic] /= 1000 / rates[topic]
    labels[topic] = f"{topic}\n({rates[topic]} Hz)"
ax2.boxplot(topics_sorted.values(), labels=labels.values())
ax2.set_ylabel("Time delay/message period")
ax2.legend()

plt.show()
