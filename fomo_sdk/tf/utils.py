import json
import os
from enum import Enum
from typing import List

import numpy as np
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
from scipy.spatial.transform import Rotation

TF_FILE_PATH = "data/calib/transforms.json"


class Format(Enum):
    MATRIX = "matrix"
    QUATERNION = "quaternion"
    JSON = "json"


class FoMoTFTree:
    def __init__(self, filepath: str = TF_FILE_PATH):
        self.tm = TransformManager()
        self.load_transforms(filepath)

    def load_transforms(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Transform file '{filepath}' not found")
        with open(filepath, "r") as f:
            transforms = json.load(f)
        for tf in transforms:
            self.tm.add_transform(tf["from"], tf["to"], self.load_tf(tf))

    def add_transform(self, from_frame: str, to_frame: str, tf: np.ndarray):
        assert tf.shape == (4, 4), "Transform must be a 4x4 matrix"
        self.tm.add_transform(from_frame, to_frame, tf)

    def visualize(self, frame: str = "base_link", whitelist: List[str] = None):
        ax = self.tm.plot_frames_in(frame, s=0.1, whitelist=whitelist)
        self.tm.plot_connections_in(frame, ax=ax, alpha=0.1, whitelist=whitelist)

    def load_tf(self, entry):
        t = np.array(
            [entry["position"]["x"], entry["position"]["y"], entry["position"]["z"]]
        )
        q = np.array(
            [
                entry["orientation"]["w"],
                entry["orientation"]["x"],
                entry["orientation"]["y"],
                entry["orientation"]["z"],
            ]
        )
        tf = pt.transform_from_pq(np.hstack((t, q)))
        # we need this since pytransform3d assumes the tf is in the to_frame coordinate system,
        # while us (and ROS) expects the from (header) coordinate system
        tf = np.linalg.inv(tf)
        return tf

    def get_transform(
        self, from_frame: str, to_frame: str, format: Format = Format.MATRIX
    ):
        """
        Returns a transformation from 'from_node' to 'to_node' in the given format.
        """
        try:
            transform = self.tm.get_transform(from_frame, to_frame)
            # we need this since pytransform3d assumes the tf is in the to_frame coordinate system,
            # while us (and ROS) expects the from (header) coordinate system
            transform = np.linalg.inv(transform)
            if format == Format.MATRIX:
                return transform
            elif format == Format.JSON or format == Format.QUATERNION:
                t = transform[:3, 3]
                q = Rotation.from_matrix(transform[:3, :3]).as_quat()
                if format == Format.QUATERNION:
                    return np.hstack((t, q))
                else:
                    output = {}
                    if len(to_frame) > 0:
                        output["to"] = to_frame
                    if len(from_frame) > 0:
                        output["from"] = from_frame
                    output["position"] = {"x": t[0], "y": t[1], "z": t[2]}
                    output["orientation"] = {"x": q[0], "y": q[1], "z": q[2], "w": q[3]}
                return json.dumps(output)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except KeyError:
            raise ValueError(f"No transform found from '{from_frame}' to '{to_frame}'")
