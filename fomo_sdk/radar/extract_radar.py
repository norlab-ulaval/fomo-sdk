from rosbags.rosbag2 import Reader
from fomo_sdk.common.fomo_mcap_writer import Writer
import fomo_sdk.common.utils as utils

input_bag_path = "/Volumes/SSD_Matej/fomo_GA/radar_test_2024_10_18-14_44_45"
output_bag_path = "radar"

# Topics to keep
topics_to_keep = ["/radar/b_scan_msg"]

typestore = utils.get_fomo_typestore()
# Open the input bag
with Reader(input_bag_path) as reader:
    # Create a new bag to write filtered data
    with Writer(output_bag_path, version=8) as writer:
        # Copy topic information
        new_conn = None
        for connection in reader.connections:
            if connection.topic in topics_to_keep:
                new_conn = writer.add_connection(
                    connection.topic, connection.msgtype, typestore=typestore
                )

        # Copy messages for the filtered topics
        for connection, timestamp, rawdata in reader.messages():
            if new_conn is not None and connection.topic in topics_to_keep:
                writer.write(new_conn, timestamp, rawdata)
