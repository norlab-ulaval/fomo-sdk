#!/usr/bin/python3
from __future__ import print_function

import os
import numpy as np
import argparse
from datetime import datetime

from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import Imu, PointCloud2

import rosbag2_py
from rclpy.serialization import deserialize_message

sub_folders = ['ouster', 'imu', 'aeva']
changeovertime = 1627387200 * 1e9

def get_num_times(bag, topics):
    times = [t for topic, msg, t in bag.read_messages(topics)]
    return len(times)

def get_start_week(rostime, gpstime):
    start_epoch = rostime * 1e-9
    dt = datetime.fromtimestamp(start_epoch)
    weekday = dt.isoweekday()
    if weekday == 7:
        weekday = 0  # Sunday
    g2 = weekday * 24 * 3600 + dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    start_week = round(start_epoch - g2)
    hour_offset = round((gpstime - g2) / 3600)
    time_zone_offset = hour_offset * 3600.0        # Toronto time is GMT-4 or GMT-5 depending on time of year
    print('START WEEK: {} TIME ZONE OFFSET: {}'.format(start_week, time_zone_offset))
    return start_week, time_zone_offset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/data1/2020_12_01/', type=str,
                        help='location of root folder. Rosbags are located under root+rosbags')
    args = parser.parse_args()
    root = args.root

    # Initialize folder structure if not done already
    for sf in sub_folders:
        if not os.path.isdir(root + sf):
            os.mkdir(root + sf)

    files = os.listdir(root)
    bagfiles = []
    for file in files:
        if file.split('.')[-1] == 'mcap':
            bagfiles.append(file)
    bagfiles.sort()

    ousterimufile = open(root + "imu/ouster_imu.csv", "w")
    ousterimufile.write('time,wx,wy,wz,ax,ay,az\n')
    aevaimufile = open(root + "imu/aeva_imu.csv", "w")
    aevaimufile.write('time,wx,wy,wz,ax,ay,az\n')

    topics = ['/ouster/points', '/ouster/imu', '/aeva/sensor/point_cloud_compensated', '/aeva/sensor/point_cloud', '/aeva/sensor/imu']

    for i in range(len(bagfiles)):
        storage_options = rosbag2_py.StorageOptions(uri=root + bagfiles[i], storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        bag = rosbag2_py.SequentialReader()
        bag.open(storage_options, converter_options)
        # num_messages = get_num_times(bag, topics)
        print('Extracting Bag {}/{} ...'.format(i + 1, len(bagfiles)))
        #tq = tqdm(total=num_messages, desc='Extracting Bag {}/{}'.format(i + 1, len(bagfiles)))
        while bag.has_next():
            topic, msg, t = bag.read_next()
            #tq.update(1) 
            if topic == '/ouster/imu' or topic == '/aeva/sensor/imu':
                imu = deserialize_message(msg, Imu)
                timestamp = imu.header.stamp.sec * 1e9 + imu.header.stamp.nanosec

            elif topic == '/ouster/points' or topic.startswith('/aeva/sensor/point_cloud'):
                msg = deserialize_message(msg, PointCloud2)
                timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
                
            if topic == '/ouster/points':
                cloud_points = list(point_cloud2.read_points(
                    msg, field_names=('x', 'y', 'z', 'intensity', 't', 'reflectivity', 'ring', 'ambient', 'range'), skip_nans=True))
                points = np.array(cloud_points, dtype=[
                    ('x', np.float32), ('y', np.float32), ('z', np.float32),
                    ('intensity', np.float32), ('t', np.float32), ('reflectivity', np.float32),
                    ('ring', np.float32), ('ambient', np.float32), ('range', np.float32)])
                timestamp = int(timestamp / 1000)
                points.tofile(root + 'ouster/{}.bin'.format(timestamp))

            if topic == '/ouster/imu':
                timestamp = int(timestamp / 1000)
                ousterimufile.write('{},{},{},{},{},{},{}\n'.format(timestamp,
                    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
                    imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))

            if topic == '/aeva/sensor/point_cloud_compensated' or topic == '/aeva/sensor/point_cloud':
                cloud_points = list(point_cloud2.read_points(
                    msg, field_names=('x', 'y', 'z', 'velocity', 'intensity', 'signal_quality', 
                                      'reflectivity', 'time_offset_ns', 'point_flags_lsb', 'point_flags_msb'), skip_nans=True))
                points = np.array(cloud_points, dtype=[
                    ('x', np.float32), ('y', np.float32), ('z', np.float32), 
                    ('velocity', np.float32), ('intensity', np.float32), ('signal_quality', np.float32), 
                    ('reflectivity', np.float32), ('time_offset_ns', np.float32), 
                    ('point_flags_lsb', np.float32), ('point_flags_msb', np.float32)])
                timestamp = int(timestamp / 1000)
                points.tofile(root + 'aeva/{}.bin'.format(timestamp))
                
            if topic == '/aeva/sensor/imu':
                timestamp = int(timestamp / 1000)
                aevaimufile.write('{},{},{},{},{},{},{}\n'.format(timestamp,
                    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
                    imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z))
    
    aevaimufile.close()
    ousterimufile.close()