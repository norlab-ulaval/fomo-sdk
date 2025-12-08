import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from fomo_sdk.tf.utils import FoMoTFTree, Format

np.set_printoptions(suppress=True)

tf_tree = FoMoTFTree()
tf_tree.add_transform(
    "vectornav_calib",
    "zedx_left",
    np.array(
        [
            [
                0.9999022975053817,
                0.013078089380118547,
                -0.0049354859562072105,
                -0.06927926773327224,
            ],
            [
                -0.0005538385613034057,
                -0.3157368528403199,
                -0.9488466330347268,
                0.4414898523924328,
            ],
            [
                -0.013967415877903167,
                0.9487566618141109,
                -0.3156987614119476,
                -0.8032729275083759,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
)

T_vectornav_to_cam = tf_tree.get_transform("vectornav_calib", "zedx_left")
T_vectornav_to_cam_cad = tf_tree.get_transform("vectornav", "zedx_left")
print(f"====vectornav new:\nnorm: {np.linalg.norm(T_vectornav_to_cam[0:3, 3])}")
rot = R.from_matrix(T_vectornav_to_cam[:3, :3])
print(f"rot:\n{rot.as_euler('xyz', degrees=True)}")
print(f"--vectornav cad\nnorm: {np.linalg.norm(T_vectornav_to_cam_cad[0:3, 3])}")
rot = R.from_matrix(T_vectornav_to_cam_cad[:3, :3])
print(f"rot:\n{rot.as_euler('xyz', degrees=True)}")

T_robosense_to_cam = tf_tree.get_transform("robosense", "zedx_left")
T_robosense_to_vectornav = np.linalg.inv(
    np.linalg.inv(T_robosense_to_cam) @ T_vectornav_to_cam
)

tf_tree.add_transform(
    "xsens_calib",
    "zedx_left",
    np.array(
        [
            [
                0.01778483694960197,
                -0.9998175080693577,
                -0.006974964706386366,
                -0.06598039128571256,
            ],
            [
                -0.3129946897579952,
                0.0010581362052458534,
                -0.9497542863978381,
                0.46714202283977896,
            ],
            [
                0.949588344367164,
                0.019074352040119535,
                -0.31291875197289987,
                -0.8801786355659601,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
)
T_xsens_to_cam = tf_tree.get_transform("xsens_calib", "zedx_left")
T_xsens_to_cam_cad = tf_tree.get_transform("xsens", "zedx_left")

print(f"====xsens new\nnorm: {np.linalg.norm(T_xsens_to_cam[0:3, 3])}")
rot = R.from_matrix(T_xsens_to_cam[:3, :3])
print(f"rot:\n{rot.as_euler('xyz', degrees=True)}")
print(f"--xsens cad\nnorm: {np.linalg.norm(T_xsens_to_cam_cad[0:3, 3])}")
rot = R.from_matrix(T_xsens_to_cam_cad[:3, :3])
print(f"rot:\n{rot.as_euler('xyz', degrees=True)}")

T_robosense_to_xsens = np.linalg.inv(
    np.linalg.inv(T_robosense_to_cam) @ T_vectornav_to_cam
)
print(tf_tree.get_transform("zedx_left", "vectornav_calib", format=Format.JSON))
print(tf_tree.get_transform("zedx_left", "xsens_calib", format=Format.JSON))
tf_tree.visualize()
plt.show()
