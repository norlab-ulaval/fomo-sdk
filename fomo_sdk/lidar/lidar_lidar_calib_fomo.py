#!/usr/bin/env python3
"""
Lidar-to-Lidar Calibration Script

This script reads lidar data from an MCAP rosbag file and performs
calibration between multiple lidar sensors using detected objects.
"""

import argparse
import copy
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from fomo_sdk.tf.utils import FoMoTFTree

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError:
    logger.error(
        "MCAP libraries not found. Please install: pip install mcap mcap-ros2-support"
    )
    exit(1)

try:
    import open3d as o3d
except ImportError:
    logger.warning("Open3D not found. Point cloud visualization will be disabled.")
    o3d = None


class LidarCalibrator:
    """Main class for lidar-to-lidar calibration"""

    def __init__(self, mcap_file: str):
        self.mcap_file = Path(mcap_file)
        self.point_clouds = {}

    def read_mcap_data(self):
        """Read lidar data from MCAP file"""
        logger.info(f"Reading MCAP file: {self.mcap_file}")

        try:
            # Read messages using the correct MCAP API
            message_count = 0

            for ros_msg in read_ros2_messages(self.mcap_file):
                message_count += 1

                # Directly access attributes of McapROS2Message
                msg = ros_msg.ros_msg
                topic = ros_msg.channel.topic
                timestamp = ros_msg.log_time_ns

                # Process different message types
                if hasattr(msg, "points") or hasattr(msg, "data"):
                    # PointCloud2 message
                    self.process_pointcloud_message(topic, msg, timestamp)

                if message_count % 100 == 0:
                    logger.info(f"Processed {message_count} messages")

            logger.info(f"Finished reading MCAP file. Total messages: {message_count}")

            if message_count == 0:
                logger.warning("No messages found in MCAP file.")

        except Exception as e:
            logger.error(f"Error reading MCAP file: {e}")

    def extract_points_numpy(
        self, msg, fields=("x", "y", "z", "intensity")
    ) -> np.ndarray:
        """Robust PointCloud2 -> numpy extraction honoring field offsets & cleaning.

        Notes:
        - Previous implementation ignored PointField.offset & point_step which caused
          misaligned reads and spurious NaNs. Those NaNs later triggered numpy's
          percentile lerp warnings (invalid value in subtract/multiply).
        - We construct a structured dtype with explicit offsets and total itemsize
          equal to msg.point_step so padding is respected.
        - We then vectorize selection of requested fields & drop rows containing NaNs.
        """
        try:
            # Fast path: use sensor_msgs_py if available (it already handles offsets)
            try:
                import sensor_msgs_py.point_cloud2 as pc2  # type: ignore

                pts_list = []
                for p in pc2.read_points(msg, field_names=fields, skip_nans=False):
                    pts_list.append(p)
                pts = np.asarray(pts_list, dtype=np.float32)
            except Exception:
                # Manual construction honoring offsets
                type_mappings = {
                    1: np.int8,
                    2: np.uint8,
                    3: np.int16,
                    4: np.uint16,
                    5: np.int32,
                    6: np.uint32,
                    7: np.float32,
                    8: np.float64,
                }

                names = []
                fmts = []
                offsets = []
                for f in msg.fields:
                    if f.datatype not in type_mappings:
                        logger.warning(
                            f"Unsupported PointField datatype {f.datatype} for field {getattr(f, 'name', '?')}, skipping"
                        )
                        continue
                    names.append(f.name)
                    fmts.append(type_mappings[f.datatype])
                    offsets.append(f.offset)

                if not names:
                    return np.empty((0, len(fields)), dtype=np.float32)

                dtype = np.dtype(
                    {
                        "names": names,
                        "formats": fmts,
                        "offsets": offsets,
                        "itemsize": msg.point_step,
                    }
                )

                raw = np.frombuffer(msg.data, dtype=dtype)
                available = [f for f in fields if f in raw.dtype.names]
                missing = [f for f in fields if f not in raw.dtype.names]
                if missing:
                    logger.debug(
                        f"PointCloud missing fields {missing}; proceeding with {available}"
                    )
                if not available:
                    return np.empty((0, len(fields)), dtype=np.float32)
                pts = np.column_stack(
                    [raw[f].astype(np.float32, copy=False) for f in available]
                )
                # If some requested fields missing, pad with zeros to keep shape consistent
                if len(available) < len(fields):
                    pad_cols = np.zeros(
                        (pts.shape[0], len(fields) - len(available)), dtype=np.float32
                    )
                    pts = np.hstack([pts, pad_cols])

            # Clean: remove rows with any NaNs or infs
            if pts.size == 0:
                return pts
            mask_finite = np.all(np.isfinite(pts[:, :3]), axis=1)
            removed = np.count_nonzero(~mask_finite)
            if removed:
                logger.debug(
                    f"Filtered out {removed} invalid points (NaN/Inf) out of {pts.shape[0]}"
                )
            pts = pts[mask_finite]

            return pts
        except Exception as e:
            logger.warning(f"Point extraction failed ({e}); returning empty array")
            return np.empty((0, len(fields)), dtype=np.float32)

    def process_pointcloud_message(self, topic: str, message, timestamp):
        """Process PointCloud2 messages"""

        # Extract point cloud data
        points = self.extract_points_numpy(message, fields=("x", "y", "z", "intensity"))

        if topic not in self.point_clouds:
            self.point_clouds[topic] = []

        # Convert timestamp from nanoseconds to seconds if needed
        time_sec = timestamp / 1e9 if timestamp > 1e10 else timestamp

        self.point_clouds[topic].append(
            {
                "timestamp": time_sec,
                "frame_id": getattr(message.header, "frame_id", f"frame_{topic}")
                if hasattr(message, "header")
                else f"frame_{topic}",
                "points": points,
            }
        )

    def detect_ring_target(self, points: np.ndarray, params: Dict) -> Optional[Dict]:
        """Jointly fit concentric ring target with known inner/outer radii.

        Assumptions:
        - Target consists of (at least) two rings: inner_radius, outer_radius.
        - Optionally include a mid radius = (inner+outer)/2 to reduce ambiguity.
        - Rings share a common center & plane normal.
        - Provided radii are accurate; we estimate center (cx, cy) in plane + normal.

        Approach:
        1. Plane estimation: PCA on candidate points (after optional pre-filter by distance).
        2. Project to 2D plane coordinates.
        3. Optimize center (u,v) on plane minimizing robust loss between each point's radial distance
           and the nearest of the known radii set R = {r_inner, r_mid, r_outer}.
        4. Compute per-ring inlier statistics and confidence.

        Returns dict with center (3D), normal, per-ring counts, confidence.
        """
        if points.shape[0] < 30:
            return None

        r_inner = float(params.get("radius_inner"))
        r_outer = float(params.get("radius_outer"))
        center_guess = np.array(
            [
                float(params.get("x_offset", 0.0)),
                float(params.get("y_offset", 0.0)),
                float(params.get("z_offset", 0.0)),
            ]
        )
        if r_inner <= 0 or r_outer <= 0 or r_inner >= r_outer:
            logger.warning("Invalid ring radii supplied; aborting ring fit")
            return None
        use_mid = params.get("use_mid_radius", True)
        r_mid = 0.5 * (r_inner + r_outer)
        radii = (
            np.array([r_inner, r_mid, r_outer])
            if use_mid
            else np.array([r_inner, r_outer])
        )

        # 1. Plane estimation via PCA (biased toward center_guess region)
        try:
            # Only use points within 4*r_outer of center_guess for PCA
            dists = np.linalg.norm(points[:, :3] - center_guess, axis=1)
            mask = dists < 4 * r_outer
            filtered_points = points[mask]
            if filtered_points.shape[0] < 10:
                raise ValueError("Not enough points near center_guess for PCA")
            pca = PCA(n_components=3)
            pca.fit(filtered_points[:, :3])
            normal = pca.components_[-1]
            # Ensure normal orientation stability (e.g., z-positive preference)
            if normal[2] < 0:
                normal = -normal
            plane_point = np.mean(filtered_points[:, :3], axis=0)
        except Exception as e:
            logger.warning(f"PCA plane fit failed: {e}")
            return None

        # Orthonormal basis in plane
        # Choose axis not parallel to normal
        ref = (
            np.array([1.0, 0.0, 0.0])
            if abs(normal[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = np.cross(normal, ref)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            ref = np.array([0.0, 1.0, 0.0])
            u = np.cross(normal, ref)
            nu = np.linalg.norm(u)
            if nu < 1e-9:
                return None
        u /= nu
        v = np.cross(normal, u)
        v /= np.linalg.norm(v)

        # Project points onto plane basis
        rel = points[:, :3] - plane_point
        pts_2d = np.stack([rel @ u, rel @ v], axis=1)
        # Initial 2D guess (project center_guess onto plane basis)
        guess_rel = center_guess - plane_point
        guess_2d = np.array([guess_rel @ u, guess_rel @ v])

        # 2/3. Optimize center shift (du,dv) to minimize robust residual to nearest ring radius
        def loss(center_shift):
            du, dv = center_shift
            shifted = pts_2d - np.array([du, dv])
            r = np.linalg.norm(shifted, axis=1)
            diff = np.min(np.abs(r[:, None] - radii[None, :]), axis=1)
            delta = params.get("huber_delta", 0.02)
            mask = diff < delta
            huber = np.where(
                mask, 0.5 * diff**2 / max(delta, 1e-6), delta * (diff - 0.5 * delta)
            )
            return np.sum(huber)

        # Simple coarse grid search + local refinement (gradient-free)
        base_extent = params.get("center_search_extent", 0.2)
        search_extent = base_extent
        prior_w = params.get("center_prior_weight", 0.0)
        if prior_w > 0:
            search_extent *= max(0.2, 1.0 - 0.6 * prior_w)
        grid = np.linspace(-search_extent, search_extent, 7)
        best_val = np.inf
        best_shift = guess_2d.copy()
        for gx in grid:
            for gy in grid:
                candidate = guess_2d + np.array([gx, gy])
                val = loss(candidate)
                if prior_w > 0:
                    val += prior_w * np.sum((candidate - guess_2d) ** 2)
                if val < best_val:
                    best_val = val
                    best_shift = candidate

        # Local random refinement
        for _ in range(50):
            proposal = best_shift + 0.02 * np.random.randn(2)
            if np.linalg.norm(proposal - guess_2d) > 2 * search_extent:
                continue
            val = loss(proposal)
            if prior_w > 0:
                val += prior_w * np.sum((proposal - guess_2d) ** 2)
            if val < best_val:
                best_val = val
                best_shift = proposal

        # Compute final assignment to rings
        shifted = pts_2d - best_shift
        r_final = np.linalg.norm(shifted, axis=1)
        nearest_idx = np.argmin(np.abs(r_final[:, None] - radii[None, :]), axis=1)
        residuals = np.abs(r_final - radii[nearest_idx])
        tol = params.get("inlier_tolerance", 0.03)
        inliers = residuals < tol
        ring_counts = {
            float(radii[i]): int(np.sum((nearest_idx == i) & inliers))
            for i in range(len(radii))
        }
        total_inliers = int(np.sum(inliers))
        confidence = total_inliers / max(points.shape[0], 1)

        # 4. Reconstruct 3D center
        center_plane = plane_point + best_shift[0] * u + best_shift[1] * v
        center_prior_error = float(np.linalg.norm(best_shift - guess_2d))

        return {
            "type": "ring_target",
            "center": center_plane.tolist(),
            "normal": normal.tolist(),
            "radii": radii.tolist(),
            "ring_counts": ring_counts,
            "inliers": total_inliers,
            "confidence": confidence,
            "residual_mean": float(np.mean(residuals[inliers]))
            if total_inliers
            else None,
            "center_prior_error": center_prior_error,
            "raw_plane_point": plane_point.tolist(),
        }

    def icp_refine_ring_target(
        self, points: np.ndarray, initial: Dict, params: Dict
    ) -> Optional[Dict]:
        """Refine ring target pose using Open3D point-to-point ICP against a synthetic concentric ring model.

        Steps:
            1. Build synthetic model point cloud consisting of sampled points on each ring (in its plane).
            2. Extract candidate scene points near the current radii (band-pass filter).
            3. Run ICP (point-to-point) to align model to scene.
            4. Update center & normal from resulting transformation.

        Notes:
            - ICP here assumes near planarity; we generate model in local plane and map to world.
            - Normal update is approximated by rotation part of ICP transform.
            - Requires Open3D; skips gracefully if unavailable or insufficient points.
        """
        if o3d is None:
            logger.debug("Open3D not available; skipping ICP refinement")
            return initial
        if initial is None:
            return None
        try:
            radii = initial.get("radii")
            if radii is None or len(radii) == 0:
                return initial
            center = initial["center"]
            normal = initial["normal"]
            if np.linalg.norm(normal) < 1e-9:
                return initial
            normal /= np.linalg.norm(normal)

            # Build basis (u,v) for plane
            ref = (
                np.array([1.0, 0.0, 0.0])
                if abs(normal[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            u = np.cross(normal, ref)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            v /= np.linalg.norm(v)

            pts_per_ring = int(params.get("icp_points_per_ring", 360))
            model_pts = []
            theta = np.linspace(0, 2 * np.pi, pts_per_ring, endpoint=False)
            for r in radii:
                ring_xy = np.column_stack([np.cos(theta) * r, np.sin(theta) * r])
                ring_xyz = center + ring_xy[:, 0, None] * u + ring_xy[:, 1, None] * v
                model_pts.append(ring_xyz)
            model_pts = np.vstack(model_pts)

            # Filter scene points near any ring radius (band-pass)
            max_ring = max(radii)
            # Crop to 4 * largest radius sphere around center
            d_center = np.linalg.norm(points[:, :3] - center, axis=1)
            mask_crop = d_center < 4 * max_ring
            scene_pts = points[mask_crop, :3]
            if scene_pts.shape[0] < 20:
                return initial
            # Project to plane to compute radial distance
            rel = scene_pts - center
            x = rel @ u
            y = rel @ v
            rho = np.sqrt(x**2 + y**2)
            band = params.get("icp_radial_band", 0.06)
            ring_mask = np.zeros_like(rho, dtype=bool)
            for r in radii:
                ring_mask |= np.abs(rho - r) < band
            scene_ring_pts = scene_pts[ring_mask]
            if scene_ring_pts.shape[0] < 30:
                logger.debug("ICP: insufficient ring points after band-pass")
                return initial

            # Construct Open3D point clouds
            model_pcd = o3d.geometry.PointCloud()
            model_pcd.points = o3d.utility.Vector3dVector(model_pts)
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(scene_ring_pts)

            # Initial guess transform = identity (model already in world frame)
            threshold = float(params.get("icp_distance_threshold", 0.05))
            max_iters = int(params.get("icp_max_iters", 30))
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iters
            )
            reg = o3d.pipelines.registration.registration_icp(
                model_pcd,
                scene_pcd,
                threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria,
            )
            if reg.transformation is None:
                return initial
            T = reg.transformation
            R_icp = T[:3, :3]
            t_icp = T[:3, 3]
            # Update center & normal (rotate basis & center, then translate)
            center_new = (R_icp @ center) + t_icp
            normal_new = R_icp @ normal
            if normal_new[2] < 0:
                normal_new = -normal_new
            normal_new /= np.linalg.norm(normal_new)
            initial["center_icp"] = center_new.tolist()
            initial["normal_icp"] = normal_new.tolist()
            initial["icp_fitness"] = reg.fitness
            initial["icp_rmse"] = reg.inlier_rmse
            return initial
        except Exception as e:
            logger.debug(f"ICP refinement failed: {e}")
            return initial


def main(mcap_file: str, transforms_file: str, lidar_frame: str):
    """Main function
    Args:
        mcap_file (str): Path to the MCAP file.
        transforms_filepath (str): Path to the transforms file. Either "config/calibration/robosense-basler/transforms.json"  # or data/calib/transforms.json
        target_lidar_frame (str): Target lidar frame. Either "hesai" or "leishen"
    """

    tf_tree = FoMoTFTree(transforms_file)
    tf_tree.visualize(frame="robosense")
    plt.show()
    # Check if files exist
    if not os.path.exists(mcap_file):
        logger.error(f"MCAP file not found: {mcap_file}")
        return

    try:
        # Create calibrator and run calibration
        calibrator = LidarCalibrator(mcap_file)
        calibrator.read_mcap_data()

        original_point_clouds = copy.deepcopy(calibrator.point_clouds)

        for topic, clouds in calibrator.point_clouds.items():
            logger.info(f"Topic: {topic}, Number of point clouds: {len(clouds)}")

        # for each point cloud in each topic, only keep the top 1% of points by intensity
        for topic, clouds in calibrator.point_clouds.items():
            for cloud in clouds:
                points = cloud["points"]
                if points.shape[1] >= 4:
                    # Use 99th percentile (top 1%) & guard against NaNs
                    valid_intensity = points[:, 3]
                    if np.any(~np.isfinite(valid_intensity)):
                        logger.debug(
                            "Intensity channel contains NaNs/Inf; filtering before percentile computation"
                        )
                        valid_mask = np.isfinite(valid_intensity)
                        valid_intensity = valid_intensity[valid_mask]
                        points = points[valid_mask]
                    if valid_intensity.size == 0:
                        continue
                    intensity_threshold = np.percentile(valid_intensity, 99)
                    cloud["points"] = points[points[:, 3] >= intensity_threshold]
        # For each point cloud in each topic, look at the remaining points after the above filtering
        # and try to detect a high intensity ring/circle using RANSAC and filtering
        # Keep the midpoint of the circle and save it to a list (keep separate lists per topic)
        circle_centers_by_topic = {topic: [] for topic in calibrator.point_clouds}
        circle_params = {
            "radius_outer": 0.5 * 0.79,
            "radius_inner": 0.5 * 0.69,
            "use_mid_radius": True,
            "center_search_extent": 0.15,
            "huber_delta": 0.02,
            # ICP params
            "icp_enable": True,
            "icp_points_per_ring": 360,
            "icp_distance_threshold": 0.5,
            "icp_max_iters": 100,
            "icp_radial_band": 0.1,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "z_offset": 0.0,
        }
        for topic, clouds in calibrator.point_clouds.items():
            for idx, cloud in enumerate(clouds):
                if idx % 100 == 0:
                    logger.info(
                        f"Processing topic '{topic}', cloud {idx + 1}/{len(clouds)}"
                    )
                points = cloud["points"]
                # Find the main cluster of remaining points using DBSCAN and compute the approximate centroid
                if len(points) > 10:
                    clustering = DBSCAN(eps=0.8, min_samples=50).fit(points[:, :3])
                    labels, counts = np.unique(clustering.labels_, return_counts=True)
                    # Filter clusters where all points are within 1 meter of each other
                    valid_clusters = []
                    for label in labels:
                        if label == -1:
                            continue
                        cluster_points = points[clustering.labels_ == label][:, :3]
                        if len(cluster_points) < 2:
                            continue
                        max_dist = np.max(
                            np.linalg.norm(
                                cluster_points - cluster_points[None, :, :], axis=2
                            )
                        )
                        if max_dist < 1.0:
                            valid_clusters.append((label, len(cluster_points)))
                    if valid_clusters:
                        # Choose the largest valid cluster
                        main_cluster_label = max(valid_clusters, key=lambda x: x[1])[0]
                        main_cluster_points = points[
                            clustering.labels_ == main_cluster_label
                        ]
                        centroid = np.mean(main_cluster_points[:, :3], axis=0)
                        circle_params["x_offset"] = centroid[0]
                        circle_params["y_offset"] = centroid[1]
                        circle_params["z_offset"] = centroid[2]
                result = calibrator.detect_ring_target(points, circle_params)
                if result is None:
                    logger.warning(
                        f"Ring target detection failed for topic '{topic}', cloud {idx + 1}/{len(clouds)}"
                    )
                    continue
                # ICP refinement (optional)
                if circle_params.get("icp_enable", False):
                    result = calibrator.icp_refine_ring_target(
                        points, result, circle_params
                    )

                params = {
                    "center_icp": result.get("center_icp", None),
                    "normal_icp": result.get("normal_icp", None),
                    "center": result["center"],
                    "normal": result["normal"],
                }

                if (
                    result is not None
                    and result.get("icp_fitness", 0) > 0.999
                    and result.get("icp_rmse", 1) < 0.05
                    and np.linalg.norm(
                        np.array(result.get("center_icp", result["center"]))
                    )
                    < 6.0
                ):
                    circle_centers_by_topic[topic].append(params)
                    # Visualize the point cloud, the detected DBSCAN centroid and the detected circle if Open3D is available
                    if o3d is not None and False:
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(points[:, :3])
                        pc.paint_uniform_color([0.6, 0.6, 0.6])

                        vis_objs = [pc]

                        # Visualize DBSCAN centroid as a small red sphere
                        # if 'x_offset' in circle_params:
                        #    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                        #    centroid_sphere.translate([circle_params['x_offset'], circle_params['y_offset'], circle_params['z_offset']])
                        #    centroid_sphere.paint_uniform_color([1, 0, 0])
                        #    vis_objs.append(centroid_sphere)

                        # Visualize ICP result if available, otherwise use refined or initial
                        if "center_icp" in result and "normal_icp" in result:
                            center = np.array(result["center_icp"])
                            normal = np.array(result["normal_icp"])
                        else:
                            center = np.array(result["center"])
                            normal = np.array(result["normal"])
                        radii = result["radii"]
                        theta = np.linspace(0, 2 * np.pi, 120)
                        # Basis
                        if np.allclose(normal, [0, 0, 1]):
                            v1 = np.array([1, 0, 0])
                        else:
                            v1 = np.cross(normal, [0, 0, 1])
                            v1 /= np.linalg.norm(v1)
                        v2 = np.cross(normal, v1)
                        v2 /= np.linalg.norm(v2)
                        colors = [[0, 1, 0], [1, 0.5, 0], [0, 0, 1]]
                        for ridx, r in enumerate(radii):
                            ring_pts = np.array(
                                [
                                    center + r * (np.cos(t) * v1 + np.sin(t) * v2)
                                    for t in theta
                                ]
                            )
                            ring_pc = o3d.geometry.PointCloud()
                            ring_pc.points = o3d.utility.Vector3dVector(ring_pts)
                            ring_pc.paint_uniform_color(colors[ridx % len(colors)])
                            vis_objs.append(ring_pc)
                        # Visualize fitted ring center as a small blue sphere (different from DBSCAN centroid)
                        fitted_center_sphere = o3d.geometry.TriangleMesh.create_sphere(
                            radius=0.05
                        )
                        fitted_center_sphere.translate(center)
                        fitted_center_sphere.paint_uniform_color([0, 0, 1])  # Blue
                        vis_objs.append(fitted_center_sphere)

                        o3d.visualization.draw_geometries(
                            vis_objs, window_name=f"{topic} cloud {idx}"
                        )
        # For each topic, create a point cloud using the detected circle centers (using icp result)
        if o3d is not None:
            topic_pcds = []
            topic_names = []
            for idx, (topic, centers) in enumerate(circle_centers_by_topic.items()):
                if len(centers) >= 2:
                    # Extract center points (prefer ICP if available)
                    pts = []
                    for c in centers:
                        if c.get("center_icp") is not None:
                            pts.append(c["center_icp"])
                        else:
                            pts.append(c["center"])
                    pts = np.array(pts)
                    center_pcd = o3d.geometry.PointCloud()
                    center_pcd.points = o3d.utility.Vector3dVector(pts)
                    # First topic: green, second: red
                    color = [0, 1, 0] if idx == 0 else [1, 0, 0]
                    center_pcd.paint_uniform_color(color)
                    topic_pcds.append(center_pcd)
                    topic_names.append(topic)
                # Store for later use if needed
                collected_center_topics = topic_names
                # Visualize both in the same window
            if len(topic_pcds) >= 2:
                o3d.visualization.draw_geometries(
                    topic_pcds,
                    window_name="Collected Centers (Green=First, Red=Second)",
                )

        # Run a 6DoF ICP between the topic_pcds given an initial transformation matrix guess
        # Afterwards, visualize the aligned point clouds together

        init_transform = tf_tree.get_transform(
            from_frame=lidar_frame, to_frame="robosense"
        )
        if o3d is not None and len(topic_pcds) >= 2:
            source_pcd = topic_pcds[1]
            target_pcd = topic_pcds[0]
            # Initial guess: identity
            threshold = 0.05  # Distance threshold for ICP
            reg_icp = o3d.pipelines.registration.registration_icp(
                source_pcd,
                target_pcd,
                threshold,
                init_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            aligned_source = copy.deepcopy(source_pcd)
            aligned_source.transform(reg_icp.transformation)
            aligned_source_init = copy.deepcopy(source_pcd)
            aligned_source_init.transform(init_transform)
            target_pcd.paint_uniform_color([0, 1, 0])  # Green
            aligned_source.paint_uniform_color([1, 0, 0])  # Red
            aligned_source_init.paint_uniform_color([1, 0, 0])  # Red
            # Visualize aligned clouds using initial guess
            o3d.visualization.draw_geometries(
                [target_pcd, aligned_source_init],
                window_name="ICP Initial Guess Alignment",
            )
            # Visualize aligned clouds
            o3d.visualization.draw_geometries(
                [target_pcd, aligned_source], window_name="ICP Aligned Centers"
            )
            logger.info(
                f"ICP fitness: {reg_icp.fitness:.4f}, RMSE: {reg_icp.inlier_rmse:.4f}"
            )
            logger.info(f"ICP transformation:\n{reg_icp.transformation}")
            tf_tree.add_transform(
                "robosense", f"{lidar_frame}_calib", reg_icp.transformation
            )
            tf_tree.visualize(frame="robosense")
            plt.show()
            from fomo_sdk.tf.utils import Format

            print(
                tf_tree.get_transform(
                    "robosense", f"{lidar_frame}_calib", format=Format.JSON
                )
            )

        # Visualize each pair of original point clouds in sequence using Open3D, applying the found ICP transform
        # User can stop by pressing any key, and continue by pressing any key again
        if o3d is not None and len(topic_pcds) >= 2:
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window(window_name="Sequential Clouds with ICP Alignment")
            num_frames = max(
                len(original_point_clouds[collected_center_topics[0]]),
                len(original_point_clouds[collected_center_topics[1]]),
            )
            frame_idx = [0]
            paused = [False]

            def toggle_pause(vis):
                paused[0] = not paused[0]
                return False  # Don't block further key events

            # Register key callback for any key to toggle pause
            for k in range(256):
                vis.register_key_callback(k, toggle_pause)

            while frame_idx[0] < num_frames:
                vis.clear_geometries()
                for t_idx, topic in enumerate(collected_center_topics):
                    clouds = original_point_clouds[topic]
                    if frame_idx[0] < len(clouds):
                        points = clouds[frame_idx[0]]["points"]
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(points[:, :3])
                        color = [0, 1, 0] if t_idx == 0 else [1, 0, 0]
                        pc.paint_uniform_color(color)
                        if t_idx == 1:
                            pc.transform(reg_icp.transformation)
                        vis.add_geometry(pc)
                vis.poll_events()
                vis.update_renderer()
                # Wait for user input if paused
                if paused[0]:
                    while paused[0]:
                        vis.poll_events()
                        vis.update_renderer()
                        time.sleep(0.05)
                else:
                    frame_idx[0] += 1
            vis.destroy_window()
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoMo lidar calibration")
    parser.add_argument("--mcap_file", type=str, help="MCAP file path")
    parser.add_argument("--transforms_file", type=str, help="Transforms file path")
    parser.add_argument(
        "--lidar_frame",
        type=str,
        choices=["hesai", "leishen"],
        default="hesai",
        help="We search for transformation from robosense to this frame.",
    )
    args = parser.parse_args()
    main(args.mcap_file, args.transforms_file, args.lidar_frame)
