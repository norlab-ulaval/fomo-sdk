
from scipy.spatial.transform import Rotation as R
import numpy as np

dict = {
    "to": "rslidar32",
    "from": "rsairy",
    "position": {
      "x": -0.185,
      "y": 0.059,
      "z": -0.083
    },
    "orientation": {
      "x": 0.0,
      "y": -0.866,
      "z": 0.0,
      "w": 0.500
    }
  }




def print_for_urdf(dict):
    t = dict["position"]

    rotation = R.from_quat(
        [
            dict["orientation"]["x"],
            dict["orientation"]["y"],
            dict["orientation"]["z"],
            dict["orientation"]["w"],
        ]
    )
    rpy = rotation.as_euler("xyz", degrees=False)


    print("-------      tf line    ------- ")
    
    print(
        f'<origin xyz="{t["x"]} {t["y"]} {t["z"]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>'
    )
    print("------------------------------- ")
    # compute the inverse transform
    tf = np.eye(4)
    tf[0:3, 3] = [t["x"], t["y"], t["z"]]
    tf[0:3, 0:3] = rotation.as_matrix()

    tf_inv = np.linalg.inv(tf)
    t_inv = tf_inv[0:3, 3]
    
    rotation_inv = R.from_matrix(tf_inv[0:3, 0:3])
    rpy_inv = rotation_inv.as_euler("xyz", degrees=False)

    print("------- inverse tf line ------- ")
    print(
        f'<origin xyz="{t_inv[0]} {t_inv[1]} {t_inv[2]}" rpy="{rpy_inv[0]} {rpy_inv[1]} {rpy_inv[2]}"/>'
    )
    print("------------------------------- ")

print_for_urdf(dict)