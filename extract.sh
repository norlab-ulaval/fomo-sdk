#!/bin/bash

pkill roslaunch
pkill rviz
pkill roscore
sleep 3s

if [ $# -eq 1 ]
  then
    root=$1
  else
    echo 'You must specify the root directory! ex: /mnt/data1/2021_10_05/'
    exit 1
fi

cp -r ./calib $root

echo 'Extracting raw sensor data from rosbags...'
python3 extract_pointclouds/extract.py --root $root

mkdir -p $root/novatel

echo 'Extracting gps log file...'
python3 extract_gps/novatel_bag_to_log.py --root $root
