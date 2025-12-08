import numpy as np
from scipy.spatial.transform import Rotation as R


def create_transform_matrix(position, quaternion):
    """Create 4x4 transformation matrix from position and quaternion"""
    # Create rotation matrix from quaternion using scipy
    rot = R.from_quat(quaternion)  # scipy expects [x, y, z, w] format
    rotation_matrix = rot.as_matrix()

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T


def invert_transform(position, orientation):
    """Invert transformation using matrix operations with scipy Rotation"""
    # Convert input to numpy arrays
    pos = np.array([position["x"], position["y"], position["z"]])
    quat = np.array(
        [orientation["x"], orientation["y"], orientation["z"], orientation["w"]]
    )

    # Create transformation matrix
    T = create_transform_matrix(pos, quat)

    # Invert the transformation matrix
    T_inv = np.linalg.inv(T)

    # Extract inverse position
    inv_pos = T_inv[:3, 3]

    # Extract inverse rotation matrix and convert to quaternion using scipy
    inv_rot_matrix = T_inv[:3, :3]
    inv_rot = R.from_matrix(inv_rot_matrix)
    inv_quat = inv_rot.as_quat()  # returns [x, y, z, w] format

    return inv_pos, inv_quat


def print_transform(position, orientation, title="Transform"):
    """Print transform in the same format as input"""
    print(f'"{title}": {{')
    print('  "position": {')
    print(f'    "x": {position[0]},')
    print(f'    "y": {position[1]},')
    print(f'    "z": {position[2]}')
    print("  },")
    print('  "orientation": {')
    print(f'    "x": {orientation[0]},')
    print(f'    "y": {orientation[1]},')
    print(f'    "z": {orientation[2]},')
    print(f'    "w": {orientation[3]}')
    print("  }")
    print("}")


# Input transformation
input_transform = {
    "position": {"x": 0.119702, "y": -0.000191556, "z": 5.27709e-5},
    "orientation": {
        "x": 0.0013524997200016092,
        "y": 0.004252502083427449,
        "z": 0.0005769131339936735,
        "w": 0.9999898770196494,
    },
}

# Print original transform
print("Original transform:")
print_transform(
    [
        input_transform["position"]["x"],
        input_transform["position"]["y"],
        input_transform["position"]["z"],
    ],
    [
        input_transform["orientation"]["x"],
        input_transform["orientation"]["y"],
        input_transform["orientation"]["z"],
        input_transform["orientation"]["w"],
    ],
    "original",
)

print("\n" + "=" * 50 + "\n")

# Invert the transformation
inv_pos, inv_quat = invert_transform(
    input_transform["position"], input_transform["orientation"]
)

# Print inverted transform
print("Inverted transform:")
print_transform(inv_pos, inv_quat, "inverted")

print("\n" + "=" * 50 + "\n")

# Verification: applying original then inverse should give identity
print("Verification (original * inverse should be near identity):")
pos = np.array(
    [
        input_transform["position"]["x"],
        input_transform["position"]["y"],
        input_transform["position"]["z"],
    ]
)
quat = np.array(
    [
        input_transform["orientation"]["x"],
        input_transform["orientation"]["y"],
        input_transform["orientation"]["z"],
        input_transform["orientation"]["w"],
    ]
)

T_original = create_transform_matrix(pos, quat)
T_inverse = create_transform_matrix(inv_pos, inv_quat)
T_result = T_original @ T_inverse

print("T_original * T_inverse =")
print(T_result)
print(
    f"Should be close to identity matrix. Max deviation: {np.max(np.abs(T_result - np.eye(4)))}"
)
