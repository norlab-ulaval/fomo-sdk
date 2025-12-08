import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from fomo_sdk.tf.utils import FoMoTFTree, Format

# replace this with your calibration results
camera_lidar_results = np.array(
    [
        0.152150253196896,
        0.4714265504097568,
        -0.21888974829559568,
        0.5640047239365824,
        -0.5534539307273268,
        0.42818036953148697,
        -0.43846207257591874,
    ]
)

np.set_printoptions(suppress=True)
tf_tree = FoMoTFTree(filepath="config/calibration/robosense-basler/transforms.json")

tf_hesai_to_basler = np.eye(4)
tf_hesai_to_basler[:3, 3] = camera_lidar_results[:3]
rot = R.from_quat(camera_lidar_results[3:], scalar_first=False)
tf_hesai_to_basler[:3, :3] = rot.as_matrix()
tf_tree.add_transform(
    from_frame="basler_calib", to_frame="hesai_calib", tf=tf_hesai_to_basler
)

print(tf_tree.get_transform("robosense", "basler_calib", Format.JSON))
tf_tree.visualize(frame="robosense")
plt.show()
