import os
import argparse
from mcap.reader import make_reader

def get_topic_list(mcap_file):
    """ Return a list of topics stored in this mcap file """
    topic_info = {}
    with open(mcap_file, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            topic_info[channel.topic] = channel.id
    print('ROS Topics Found in file:')
    for topic in topic_info:
        print(topic)
    raw_topic_id = topic_info.get('/novatel/oem7/oem7raw', None)
    if raw_topic_id is None:
        print('\nNo Raw NovAtel Logs found in file.')
    return raw_topic_id

def get_topic_messages(mcap_file, topic, outpath):
    """
    Extract messages for a specific topic in this mcap file
    """
    out_path = os.path.join(outpath, os.path.splitext(os.path.basename(mcap_file))[0] + '.log')
    with open(mcap_file, "rb") as f, open(out_path, 'wb') as fout:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            if channel.topic == topic:
                fout.write(message.data)
    return 0

def parse_args():
    usage = "Extract raw NovAtel data from a ROS2 Bag 'mcap' file" \
            "  usage: ros2_to_raw -f <file.mcap>"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('-f', help='Absolute path to input file')
    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/data1/2020_12_01/', type=str,
                        help='location of root folder. Rosbags are located under root')
    args = parser.parse_args()
    root = args.root

    mcap_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.mcap')]
    if not mcap_files:
        print(f"No .mcap files found in {root}")
        exit(1)
    print(f"Found {len(mcap_files)} .mcap in {root}")

    for bag_file in mcap_files:
        print(f"Processing {bag_file}")
        out_path = os.path.join(root, 'novatel')
        raw_topic_id = get_topic_list(bag_file)
        if raw_topic_id:
            get_topic_messages(bag_file, topic='/novatel/oem7/oem7raw', outpath=out_path)