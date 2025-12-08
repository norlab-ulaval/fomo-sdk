#!/bin/bash

DEPLOYMENT_DIR="$HOME/dataset/FoMo/Deployment_2_21nov2024"
GNSS_BAG="$DEPLOYMENT_DIR/gnss/M2/emlid_rosbag/metadata.yaml"
ROSBAGS_DIR="$DEPLOYMENT_DIR/rosbags"

# Colors to process
COLORS=("blue" "green" "magenta" "orange" "red" "yellow")

echo "Starting times for each trajectory:"
for COLOR in "${COLORS[@]}"; do
    METADATA_FILE="$ROSBAGS_DIR/$COLOR/metadata.yaml"
    if [[ -f "$METADATA_FILE" ]]; then
        START_TIME=$(awk '/nanoseconds_since_epoch:/ {print $2; exit}' "$METADATA_FILE")
        echo "$COLOR: $START_TIME"
    else
        echo "$COLOR: metadata.yaml not found"
    fi
done

# 1. Read every rosbag trajectory metadata.yaml for each deployment day 
# 2. Divide each deployment mcap_emlid_rosbag into all the trajectories based on the timestamps
# 3. Save each trajectory as a separate mcap file in the ground_truth folder
# 4. create a ground_truth folder for each deployment day 
# 5. run the six_dof_generation.py script to generate the six_dof ground truth for each trajectory run in the ground_truth folder
