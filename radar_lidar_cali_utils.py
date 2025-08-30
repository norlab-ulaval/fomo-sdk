
import numpy as np


def project_lidar_onto_radar(points, max_elev=0.05):
    # find points that are within the radar scan FOV
    points_out = []
    for i in range(points.shape[0]):
        elev = np.arctan2(points[i, 2], np.sqrt(points[i, 0]**2 + points[i, 1]**2))
        if np.abs(elev) <= max_elev:
            points_out.append(points[i, :])
    points = np.array(points_out)
    # project to 2D (spherical projection)
    for i in range(points.shape[0]):
        rho = np.sqrt(points[i, 0]**2 + points[i, 1]**2 + points[i, 2]**2)
        phi = np.arctan2(points[i, 1], points[i, 0])
        points[i, 0] = rho * np.cos(phi)
        points[i, 1] = rho * np.sin(phi)
        points[i, 2] = 0.0
    return points


def polar_to_cartesian_points(
    azimuths: np.ndarray,
    polar_points: np.ndarray,
    radar_resolution: float,
    downsample_rate=1,
    range_offset = -0.31,
) -> np.ndarray:
    """Converts points from polar coordinates to cartesian coordinates
    Args:
        azimuths (np.ndarray): The actual azimuth of reach row in the fft data reported by the Navtech sensor
        polar_points (np.ndarray): N x 2 array of points (azimuth_bin, range_bin)
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        downsample_rate (float): fft data may be downsampled along the range dimensions to speed up computation
    Returns:
        np.ndarray: N x 2 array of points (x, y) in metric
    """
    N = polar_points.shape[0]
    cart_points = np.zeros((N, 2))
    for i in range(0, N):
        azimuth = azimuths[int(polar_points[i, 0])]
        r = polar_points[i, 1] * radar_resolution * downsample_rate + radar_resolution / 2 + range_offset
        cart_points[i, 0] = r * np.cos(azimuth)
        cart_points[i, 1] = r * np.sin(azimuth)
    return cart_points

def convert_to_bev(cart_points: np.ndarray, cart_resolution: float, cart_pixel_width: int) -> np.ndarray:
    """Converts points from metric cartesian coordinates to pixel coordinates in the BEV image
    Args:
        cart_points (np.ndarray): N x 2 array of points (x, y) in metric
        cart_pixel_width (int): width and height of the output BEV image
    Returns:
        np.ndarray: N x 2 array of points (u, v) in pixels which can be plotted on the BEV image
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    pixels = []
    N = cart_points.shape[0]
    for i in range(0, N):
        u = (cart_min_range + cart_points[i, 1]) / cart_resolution
        v = (cart_min_range - cart_points[i, 0]) / cart_resolution
        if 0 < u and u < cart_pixel_width and 0 < v and v < cart_pixel_width:
            pixels.append((u, v))
    return np.asarray(pixels)

def modifiedCACFAR(
    raw_scan: np.ndarray,
    minr=2.0,
    maxr=80.0,
    res=0.04381,
    width=101,
    guard=5,
    threshold=1.0,
    threshold2=0.0,
    threshold3=0.09,
    peak_summary_method='max_intensity'):
    # peak_summary_method: median, geometric_mean, max_intensity, weighted_mean
    rows = raw_scan.shape[0]
    cols = raw_scan.shape[1]
    if width % 2 == 0: width += 1
    w2 = int(np.floor(width / 2))
    mincol = int(minr / res + w2 + guard + 1)
    if mincol > cols or mincol < 0: mincol = 0
    maxcol = int(maxr / res - w2 - guard)
    if maxcol > cols or maxcol < 0: maxcol = cols
    N = maxcol - mincol
    targets_polar_pixels = []
    for i in range(rows):
        mean = np.mean(raw_scan[i])
        peak_points = []
        peak_point_intensities = []
        for j in range(mincol, maxcol):
            left = 0
            right = 0
            for k in range(-w2 - guard, -guard):
                left += raw_scan[i, j + k]
            for k in range(guard + 1, w2 + guard):
                right += raw_scan[i, j + k]
            # (statistic) estimate of clutter power
            stat = max(left, right) / w2  # GO-CFAR
            thres = threshold * stat + threshold2 * mean + threshold3
            if raw_scan[i, j] > thres:
                peak_points.append(j)
                peak_point_intensities.append(raw_scan[i, j])
            elif len(peak_points) > 0:
                if peak_summary_method == 'median':
                    r = peak_points[len(peak_points) // 2]
                elif peak_summary_method == 'geometric_mean':
                    r = np.mean(peak_points)
                elif peak_summary_method == 'max_intensity':
                    r = peak_points[np.argmax(peak_point_intensities)]
                elif peak_summary_method == 'weighted_mean':
                    r = np.sum(np.array(peak_points) * np.array(peak_point_intensities) / np.sum(peak_point_intensities))
                else:
                    raise NotImplementedError("peak summary method: {} not supported".format(peak_summary_method))
                targets_polar_pixels.append((i, r))
                peak_points = []
                peak_point_intensities = []
    return np.asarray(targets_polar_pixels)

def KStrong(
    raw_scan: np.ndarray,
    minr=2.0,
    maxr=80.0,
    res=0.04381,
    K=3,
    static_threshold=0.25):
    rows = raw_scan.shape[0]
    cols = raw_scan.shape[1]
    mincol = int(minr / res)
    if mincol > cols or mincol < 0: mincol = 0
    maxcol = int(maxr / res)
    if maxcol > cols or maxcol < 0: maxcol = cols
    
    targets_polar_pixels = []

    for i in range(rows):

        temp_intensities = raw_scan[i]
        max_pairs = []

        for j in range(mincol, maxcol):
            if temp_intensities[j] > static_threshold:
                max_pairs.append((temp_intensities[j], j))

        sorted_pairs = sorted(max_pairs, key=lambda x: x[0], reverse=True)

        for k in range(K):
            if k < len(sorted_pairs):
                value, j = sorted_pairs[k]
                assert(value==raw_scan[i, j])
                targets_polar_pixels.append((i, j))
            else:
                break

    return np.asarray(targets_polar_pixels)


import open3d as o3d
def icp_multistage(radar_pts, lidar_pts, T_init=None, crop_margin=2.0, verbose=True):
    """
    Multi-stage ICP to estimate T_lidar<-radar.
    Assumes:
      - radar_pts: (N,2) as [x,y]  (or (N,>=2), only first 2 used). Lifted to z=0.
      - lidar_pts: (M,3) as [x,y,z].
      - T_init:    4x4 initial guess mapping radar->lidar (e.g., from CAD). If None, identity.

    Strategy:
      1) Build radar XYZ as (x,y,0).
      2) Optional XY crop of LiDAR around transformed radar (using T_init) to boost overlap.
      3) Coarse-to-fine ICP:
           - Start point-to-point with large corr. distances.
           - Switch to point-to-plane after “contact”.
      4) Return refined T, fitness, rmse (Open3D definitions).
    """
    import numpy as np
    import open3d as o3d

    def apply_T(T, P):
        P = np.asarray(P, dtype=np.float64)
        return (P @ T[:3, :3].T) + T[:3, 3]

    def to_o3d_pcd(P, estimate_normals=False, voxel=None):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, dtype=np.float64)))
        if voxel and voxel > 0.0:
            pcd = pcd.voxel_down_sample(voxel)
        if estimate_normals:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            pcd.orient_normals_consistent_tangent_plane(k=30)
        return pcd

    radar_pts = np.asarray(radar_pts, dtype=np.float64)
    if radar_pts.ndim != 2 or radar_pts.shape[1] < 2:
        raise ValueError("radar_pts must be (N,2) or (N,>=2).")
    radar_xyz = np.c_[radar_pts[:, 0], radar_pts[:, 1], np.zeros(len(radar_pts))]
    lidar_xyz = np.asarray(lidar_pts, dtype=np.float64)
    if lidar_xyz.ndim != 2 or lidar_xyz.shape[1] != 3:
        raise ValueError("lidar_pts must be (M,3).")

    radar_xyz = radar_xyz[np.isfinite(radar_xyz).all(axis=1)]
    lidar_xyz = lidar_xyz[np.isfinite(lidar_xyz).all(axis=1)]

    T = np.eye(4, dtype=np.float64) if T_init is None else np.array(T_init, dtype=np.float64)
    if crop_margin is not None and crop_margin > 0:
        radar_in_lidar0 = apply_T(T, radar_xyz)
        rx_min, ry_min = radar_in_lidar0[:, 0].min() - crop_margin, radar_in_lidar0[:, 1].min() - crop_margin
        rx_max, ry_max = radar_in_lidar0[:, 0].max() + crop_margin, radar_in_lidar0[:, 1].max() + crop_margin
        mask = (
            (lidar_xyz[:, 0] >= rx_min) & (lidar_xyz[:, 0] <= rx_max) &
            (lidar_xyz[:, 1] >= ry_min) & (lidar_xyz[:, 1] <= ry_max)
        )
        lidar_cropped = lidar_xyz[mask]
        # if crop nuked too much, fall back
        if len(lidar_cropped) >= 1000:
            lidar_xyz = lidar_cropped
            if verbose:
                print(f"[ICP] XY-crop kept {len(lidar_xyz)} LiDAR points.")
        elif verbose:
            print("[ICP] Skipping crop (too few points kept).")

    radar_pcd = to_o3d_pcd(radar_xyz, estimate_normals=False, voxel=0.10)
    lidar_pcd = to_o3d_pcd(lidar_xyz, estimate_normals=True,  voxel=0.10)

    # Estimators
    est_pt2pt = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    est_pt2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    stages = [
        dict(voxel=0.25, max_corr_dist=1.50, iters=60, estimator=est_pt2pt),  # very coarse grab
        dict(voxel=0.15, max_corr_dist=0.80, iters=50, estimator=est_pt2pt),
        dict(voxel=0.10, max_corr_dist=0.50, iters=60, estimator=est_pt2pl),  # refine with planes
        dict(voxel=0.05, max_corr_dist=0.25, iters=80, estimator=est_pt2pl),
    ]

    for s in stages:
        # Re-voxel for this stage; apply current T to radar before ICP
        radar_stage = to_o3d_pcd(apply_T(T, np.asarray(radar_pcd.points)), estimate_normals=False, voxel=s["voxel"])
        lidar_stage = to_o3d_pcd(np.asarray(lidar_pcd.points), estimate_normals=True, voxel=s["voxel"])

        reg = o3d.pipelines.registration.registration_icp(
            radar_stage,
            lidar_stage,
            s["max_corr_dist"],
            np.eye(4),  # we already applied T to radar_stage
            s["estimator"],
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=s["iters"]),
        )
        T = reg.transformation @ T
        if verbose:
            mode = "p2p" if s["estimator"] is est_pt2pt else "p2pl"
            print(f"[ICP] stage voxel={s['voxel']:.2f} corr={s['max_corr_dist']:.2f} "
                  f"mode={mode} fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.3f}")

    radar_eval = to_o3d_pcd(apply_T(T, np.asarray(radar_pcd.points)), voxel=0.05)
    lidar_eval = to_o3d_pcd(np.asarray(lidar_pcd.points), estimate_normals=True, voxel=0.05)
    reg_eval = o3d.pipelines.registration.evaluate_registration(radar_eval, lidar_eval, 0.20)

    if verbose:
        print("\nICP Result - T_ref:\n", T)
        print("Fitness:", reg_eval.fitness, "RMSE:", reg_eval.inlier_rmse)

    return T, float(reg_eval.fitness), float(reg_eval.inlier_rmse)
