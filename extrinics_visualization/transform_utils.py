"""
Common transformation utilities for visualization scripts.
This module contains shared functions for handling SE3 transformations and coordinate conversions.
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque


def load_transform_tree(json_file_path):
    """Load the transform tree from JSON file"""
    with open(json_file_path, 'r') as f:
        return json.load(f)


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix"""
    # Create scipy rotation object from quaternion (x, y, z, w)
    rotation = R.from_quat([qx, qy, qz, qw])
    return rotation.as_matrix()


def position_to_translation(px, py, pz):
    """Convert position to translation vector"""
    return np.array([px, py, pz])


def create_se3_matrix(rotation_matrix, translation_vector):
    """Create SE3 transformation matrix from rotation matrix and translation vector"""
    se3 = np.eye(4)
    se3[:3, :3] = rotation_matrix
    se3[:3, 3] = translation_vector
    return se3


def find_transform_path(transform_tree, source_frame, target_frame):
    """Find the path from source to target frame using BFS"""
    # Build adjacency list
    graph = {}
    for transform in transform_tree:
        from_frame = transform['from']
        to_frame = transform['to']
        
        if from_frame not in graph:
            graph[from_frame] = []
        if to_frame not in graph:
            graph[to_frame] = []
            
        graph[from_frame].append((to_frame, transform, False))  # False indicates forward transform
        graph[to_frame].append((from_frame, transform, True))  # True indicates reverse transform
    
    # BFS to find path
    queue = deque([(source_frame, [])])
    visited = {source_frame}
    
    while queue:
        current_frame, path = queue.popleft()
        
        if current_frame == target_frame:
            return path
        
        for neighbor, transform_data, is_reverse in graph.get(current_frame, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [(neighbor, transform_data, is_reverse)]
                queue.append((neighbor, new_path))
    
    return None  # No path found


def get_se3_extrinsic(source_frame, target_frame, transform_json_path="transform.json"):
    """
    Get SE3 extrinsic pose from source frame to target frame
    
    Args:
        source_frame (str): Source frame name
        target_frame (str): Target frame name  
        transform_json_path (str): Path to transform.json file
        
    Returns:
        numpy.ndarray: 4x4 SE3 transformation matrix from source to target
        None: If no path found between frames
    """
    # Load transform tree
    transform_tree = load_transform_tree(transform_json_path)
    
    # Find path from source to target
    path = find_transform_path(transform_tree, source_frame, target_frame)
    
    if path is None:
        print(f"No path found from {source_frame} to {target_frame}")
        return None
    
    # Start with identity matrix
    cumulative_transform = np.eye(4)
    
    # Apply each transformation in the path
    for frame_name, transform_data, is_reverse in path:
        # Extract position and orientation
        pos = transform_data['position']
        orient = transform_data['orientation']
        
        # Convert to rotation matrix and translation vector
        rotation_matrix = quaternion_to_rotation_matrix(
            orient['x'], orient['y'], orient['z'], orient['w']
        )
        translation_vector = position_to_translation(
            pos['x'], pos['y'], pos['z']
        )
        
        # Create SE3 matrix
        se3_matrix = create_se3_matrix(rotation_matrix, translation_vector)
        
        if is_reverse:
            # For reverse transforms, we need the inverse
            se3_matrix = np.linalg.inv(se3_matrix)
        
        # Compose with cumulative transform
        cumulative_transform = cumulative_transform @ se3_matrix
    
    return cumulative_transform


def print_se3_info(se3_matrix, source_frame, target_frame):
    """Print human-readable information about the SE3 transformation"""
    if se3_matrix is None:
        print(f"Failed to find transformation from {source_frame} to {target_frame}")
        return
    
    # Extract rotation and translation
    rotation_matrix = se3_matrix[:3, :3]
    translation_vector = se3_matrix[:3, 3]
    
    # Convert rotation matrix to euler angles (for readability)
    rotation_obj = R.from_matrix(rotation_matrix)
    euler_angles = rotation_obj.as_euler('xyz', degrees=True)
    
    print(f"\nTransformation from {source_frame} to {target_frame}:")
    print(f"Translation (x, y, z): [{translation_vector[0]:.6f}, {translation_vector[1]:.6f}, {translation_vector[2]:.6f}]")
    print(f"Rotation (roll, pitch, yaw) in degrees: [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
    print(f"SE3 Matrix:")
    print(se3_matrix)
