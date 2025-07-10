#!/bin/bash

# Check if extract.sh script exists
if [ ! -e extract.sh ]; then
    echo "Error: extras.sh script not found. Make sure it exists in the current directory."
    exit 1
fi

# Define file paths manually
file_paths=(
    "/home/katya/ASRL/vtr3/data/rosbag2_2025_05_05-18_47_11/"
    "/home/katya/ASRL/vtr3/data/rosbag2_2025_05_05-18_50_14/"
)

# Check if the file paths are valid
for file_path in "${file_paths[@]}"; do
    if [ ! -d "$file_path" ]; then
        echo "Error: Directory $file_path does not exist."
        exit 1
    fi
done

# Loop through the file paths and run extract.sh for each file
for file_path in "${file_paths[@]}"; do
    echo "Running extract.sh for file: $file_path"
    ./extract.sh "$file_path"
done

echo "Finished running extract.sh for all specified files."