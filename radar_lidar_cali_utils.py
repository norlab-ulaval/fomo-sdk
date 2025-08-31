
import numpy as np
from scipy import ndimage

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

def cen2018features(fft_data: np.ndarray, min_range=58, zq=4.0, sigma_gauss=17) -> np.ndarray:
    """Extract features from polar radar data using the method described in cen_icra18
    Args:
        fft_data (np.ndarray): Polar radar power readings
        min_range (int): targets with a range bin less than or equal to this value will be ignored.
        zq (float): if y[i] > zq * sigma_q then it is considered a potential target point
        sigma_gauss (int): std dev of the gaussian filter used to smooth the radar signal
        
    Returns:
        np.ndarray: N x 2 array of feature locations (azimuth_bin, range_bin)
    """
    nazimuths = fft_data.shape[0]
    # w_median = 200
    # q = fft_data - ndimage.median_filter(fft_data, size=(1, w_median))  # N x R
    q = fft_data - np.mean(fft_data, axis=1, keepdims=True)
    p = ndimage.gaussian_filter1d(q, sigma=17, truncate=3.0) # N x R
    noise = np.where(q < 0, q, 0) # N x R
    nonzero = np.sum(q < 0, axis=-1, keepdims=True) # N x 1
    sigma_q = np.sqrt(np.sum(noise**2, axis=-1, keepdims=True) / nonzero) # N x 1

    def norm(x, sigma):
        return np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    nqp = norm(q - p, sigma_q)
    npp = norm(p, sigma_q)
    nzero = norm(np.zeros((nazimuths, 1)), sigma_q)
    y = q * (1 - nqp / nzero) + p * ((nqp - npp) / nzero)
    t = np.nonzero(y > zq * sigma_q)
    # Extract peak centers
    current_azimuth = t[0][0]
    peak_points = [t[1][0]]
    peak_centers = []

    def mid_point(l):
        return l[len(l) // 2]

    for i in range(1, len(t[0])):
        if t[1][i] - peak_points[-1] > 1 or t[0][i] != current_azimuth:
            m = mid_point(peak_points)
            if m > min_range:
                peak_centers.append((current_azimuth, m))
            peak_points = []
        current_azimuth = t[0][i]
        peak_points.append(t[1][i])
    if len(peak_points) > 0 and mid_point(peak_points) > min_range:
        peak_centers.append((current_azimuth, mid_point(peak_points)))

    return np.asarray(peak_centers)

# modified CACFAR algorithm
def modifiedCACFAR(
    raw_scan: np.ndarray,
    minr=1.0,
    maxr=69.0,
    res=0.040308,
    width=137,
    guard=7,
    threshold=0.50,
    threshold2=0.0,
    threshold3=0.23,
    peak_summary_method='weighted_mean'):
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

    # print("In Modified CACFAR: maxcol:",maxcol)
    # print("In Modified CACFAR: mincol:",mincol)

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

def KPeaks(
    raw_scan: np.ndarray,
    minr: float = 2.0,
    maxr: float = 80.0,
    res: float = 0.04381,
    K: int = 3,
    static_threshold: float = 0.25,
):
    """
    K-peaks radar extractor (Python)
    - raw_scan: 2D array [rows=azimuth, cols=range_bins] of float intensities
    - minr/maxr: meters
    - res: meters per bin (range resolution)
    - K: number of peaks to keep per row
    - static_threshold: intensity threshold to start a peak

    Returns:
      np.ndarray of shape [N, 2], entries (row_index, avg_col_index_float)
      Note: avg_col_index can be fractional due to averaging across a peak.
    """
    rows, cols = raw_scan.shape

    # convert meter limits to column limits, clamp to [0, cols]
    mincol = int(minr / res)
    if mincol > cols or mincol < 0:
        mincol = 0
    maxcol = int(maxr / res)
    if maxcol > cols or maxcol < 0:
        maxcol = cols

    targets_polar_pixels = []

    for i in range(rows):
        # 1) Collect (intensity, j) for bins above threshold in increasing j
        intens = []
        row_vals = raw_scan[i]
        for j in range(mincol, maxcol):
            v = row_vals[j]
            if v >= static_threshold:
                intens.append((v, j))

        if not intens:
            continue

        # 2) Group adjacent bins into peaks, tracking each peak’s max intensity
        peaks = []  # list of (peak_max_value, [bin_indices])
        current_bins = [intens[0][1]]
        current_max = intens[0][0]

        for val, j in intens[1:]:
            if j == current_bins[-1] + 1:
                # continue the current peak
                current_bins.append(j)
                if val > current_max:
                    current_max = val
            else:
                # finalize previous peak
                peaks.append((current_max, current_bins))
                # start new peak
                current_bins = [j]
                current_max = val

        # add the last peak
        peaks.append((current_max, current_bins))

        # 3) Sort peaks by max intensity (desc)
        peaks.sort(key=lambda x: x[0], reverse=True)

        # 4) Take top-K peaks; use averaged column index for each peak
        for p in range(min(K, len(peaks))):
            _, bins = peaks[p]
            avg_j = float(np.mean(bins))  # can be fractional
            # (i, avg_j) mirrors your KStrong (row, col) output convention
            targets_polar_pixels.append((i, avg_j))

    return np.asarray(targets_polar_pixels, dtype=np.float32)



import open3d as o3d
import matplotlib.pyplot as plt

def apply_T(T, P):
    P = np.asarray(P, dtype=np.float64)
    return (P @ T[:3, :3].T) + T[:3, 3]

def icp_multistage(radar_pts, lidar_pts, T_init=None, crop_margin=2.0, verbose=True):
    """
    Multi-stage ICP to estimate T_lidar<-radar, but constrained to x, y, yaw.
    z, roll, pitch are frozen to the prior (T_init).
    Assumes:
      - radar_pts: (N,2) [x,y]  (lifted to z=0 internally)
      - lidar_pts: (M,3) [x,y,z]
      - T_init: 4x4 mapping radar->lidar (CAD). If None, identity (z=0, r=p=0).
    """
    # ---------------- helpers ----------------
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

    def rpy_from_R_xyz(R):
        # XYZ (roll,pitch,yaw) with R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        roll  = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

    def project_xy_yaw(T_cur, T_prior):
        """Keep x,y and yaw from T_cur; keep z, roll, pitch from T_prior."""
        R_prior = T_prior[:3, :3]
        t_prior = T_prior[:3, 3]
        r0, p0, _ = rpy_from_R_xyz(R_prior)  # keep roll, pitch from prior

        R_cur = T_cur[:3, :3]
        t_cur = T_cur[:3, 3]
        _, _, y = rpy_from_R_xyz(R_cur)      # yaw from current

        R_new = Rz(y) @ Ry(p0) @ Rx(r0)      # compose with yaw current, r/p prior
        t_new = t_cur.copy()
        t_new[2] = t_prior[2]                # z from prior

        Tout = np.eye(4, dtype=np.float64)
        Tout[:3, :3] = R_new
        Tout[:3, 3] = t_new
        return Tout

    # ---------------- inputs ----------------
    radar_pts = np.asarray(radar_pts, dtype=np.float64)
    if radar_pts.ndim != 2 or radar_pts.shape[1] < 2:
        raise ValueError("radar_pts must be (N,2) or (N,>=2).")
    radar_xyz = np.c_[radar_pts[:, 0], radar_pts[:, 1], np.zeros(len(radar_pts))]

    lidar_xyz = np.asarray(lidar_pts, dtype=np.float64)
    if lidar_xyz.ndim != 2 or lidar_xyz.shape[1] != 3:
        raise ValueError("lidar_pts must be (M,3).")

    # Clean NaNs/Infs
    radar_xyz = radar_xyz[np.isfinite(radar_xyz).all(axis=1)]
    lidar_xyz = lidar_xyz[np.isfinite(lidar_xyz).all(axis=1)]

    # Prior (frozen z/r/p come from here)
    T = np.eye(4, dtype=np.float64) if T_init is None else np.array(T_init, dtype=np.float64)
    T_prior = T.copy()  # snapshot to keep z/roll/pitch

    # ---------------- crop (XY only) ----------------
    if crop_margin is not None and crop_margin > 0:
        radar_in_lidar0 = apply_T(T, radar_xyz)
        rx_min, ry_min = radar_in_lidar0[:, 0].min() - crop_margin, radar_in_lidar0[:, 1].min() - crop_margin
        rx_max, ry_max = radar_in_lidar0[:, 0].max() + crop_margin, radar_in_lidar0[:, 1].max() + crop_margin
        mask = (
            (lidar_xyz[:, 0] >= rx_min) & (lidar_xyz[:, 0] <= rx_max) &
            (lidar_xyz[:, 1] >= ry_min) & (lidar_xyz[:, 1] <= ry_max)
        )
        lidar_cropped = lidar_xyz[mask]
        if len(lidar_cropped) >= 1000:
            lidar_xyz = lidar_cropped
            if verbose:
                print(f"[ICP] XY-crop kept {len(lidar_xyz)} LiDAR points.")
        elif verbose:
            print("[ICP] Skipping crop (too few points kept).")

    # ---------------- base clouds ----------------
    radar_pcd = to_o3d_pcd(radar_xyz, estimate_normals=False, voxel=0.10)
    lidar_pcd = to_o3d_pcd(lidar_xyz, estimate_normals=True,  voxel=0.10)

    est_pt2pt = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    est_pt2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    stages = [
        dict(voxel=0.25, max_corr_dist=1.50, iters=60, estimator=est_pt2pt),  # coarse grab
        dict(voxel=0.15, max_corr_dist=0.80, iters=50, estimator=est_pt2pt),
        dict(voxel=0.10, max_corr_dist=0.50, iters=60, estimator=est_pt2pl),  # plane refine
        dict(voxel=0.05, max_corr_dist=0.25, iters=80, estimator=est_pt2pl),
        dict(voxel=0.02, max_corr_dist=0.10, iters=100, estimator=est_pt2pl),
        dict(voxel=0.01, max_corr_dist=0.05, iters=100, estimator=est_pt2pl),
    ]

    for s in stages:
        # Re-voxel for this stage; apply current T to radar before ICP
        radar_stage = to_o3d_pcd(apply_T(T, np.asarray(radar_pcd.points)), estimate_normals=False, voxel=s["voxel"])
        lidar_stage = to_o3d_pcd(np.asarray(lidar_pcd.points), estimate_normals=True, voxel=s["voxel"])

        reg = o3d.pipelines.registration.registration_icp(
            radar_stage,
            lidar_stage,
            s["max_corr_dist"],
            np.eye(4),  # incremental ICP; we've pre-applied T
            s["estimator"],
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=s["iters"]),
        )

        # Compose increment, then PROJECT to (x,y,yaw) only
        T = reg.transformation @ T
        T = project_xy_yaw(T, T_prior=T_prior)

        if verbose:
            mode = "p2p" if s["estimator"] is est_pt2pt else "p2pl"
            print(f"[ICP] stage voxel={s['voxel']:.2f} corr={s['max_corr_dist']:.2f} "
                  f"mode={mode} fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.3f}")

    # ---------------- final evaluation ----------------
    radar_eval = to_o3d_pcd(apply_T(T, np.asarray(radar_pcd.points)), voxel=0.05)
    lidar_eval = to_o3d_pcd(np.asarray(lidar_pcd.points), estimate_normals=True, voxel=0.05)
    reg_eval = o3d.pipelines.registration.evaluate_registration(radar_eval, lidar_eval, 0.20)

    if verbose:
        print("\nICP (x,y,yaw) Result - T_ref:\n", T)
        print("Fitness:", reg_eval.fitness, "RMSE:", reg_eval.inlier_rmse)

    return T, float(reg_eval.fitness), float(reg_eval.inlier_rmse)

# for multi-frame ICP where we take the mean of the output se3 pose
def se3_log(T):
    R = T[:3,:3]; t = T[:3,3]
    theta = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if theta < 1e-9:
        w = np.zeros(3); V_inv = np.eye(3)
    else:
        wx = (1/(2*np.sin(theta))) * np.array([
            R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]
        ])
        w = theta * wx
        A = np.sin(theta)/theta
        B = (1-np.cos(theta))/(theta**2)
        V_inv = np.eye(3) - 0.5*skew(w) + (1/(theta**2))*(1 - A/(2*B))*(skew(w)@skew(w))
    v = V_inv @ t
    return np.r_[v, w]

def se3_exp(xi):
    v, w = xi[:3], xi[3:]
    th = np.linalg.norm(w)
    if th < 1e-9:
        R = np.eye(3); V = np.eye(3)
    else:
        k = w/th
        K = skew(k)
        R = np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)
        A = np.sin(th)/th
        B = (1-np.cos(th))/(th**2)
        V = np.eye(3) + B*(K) + ((1-A)/ (th**2))*(K@K)
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=V@v
    return T

def skew(w): 
    return np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])

def se3_mean(T_list, iters=10):
    X = np.eye(4)
    for _ in range(iters):
        xi_sum = np.zeros(6)
        for T in T_list:
            delta = np.linalg.inv(X) @ T
            xi_sum += se3_log(delta)
        X = X @ se3_exp(xi_sum/len(T_list))
    return X

# checking alignement visualization
def to_pcd(P, color=None, voxel=None, normals=False):
    P = np.asarray(P, float)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    if color is not None:
        pcd.paint_uniform_color(color)
    if normals:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd

def visualize_alignment_3d(radar_xy, lidar_xyz, T,
                           voxel_lidar=0.05, voxel_radar=0.03, point_size=2.0,
                           margin=1.0):
    # Lift radar to z=0 and transform into LiDAR frame
    radar_xyz = np.c_[np.asarray(radar_xy, float)[:, 0],
                      np.asarray(radar_xy, float)[:, 1],
                      np.zeros(len(radar_xy))]
    radar_L = apply_T(np.asarray(T, float), radar_xyz)

    # Build colored point clouds (optional downsample for speed)
    lidar_pcd = to_pcd(lidar_xyz, color=[0.6, 0.6, 0.6], voxel=voxel_lidar)
    radar_pcd = to_pcd(radar_L,    color=[1.0, 0.1, 0.1], voxel=voxel_radar)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Radar→LiDAR Alignment", width=1280, height=800, visible=True)
    for g in [lidar_pcd, radar_pcd, axes]:
        vis.add_geometry(g)

    # Rendering options
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.array([1, 1, 1])  # white bg

    # Compute a combined AABB and inflate it manually (no .expand())
    aabb_L = lidar_pcd.get_axis_aligned_bounding_box()
    aabb_R = radar_pcd.get_axis_aligned_bounding_box()
    minb = np.minimum(aabb_L.get_min_bound(), aabb_R.get_min_bound()) - margin
    maxb = np.maximum(aabb_L.get_max_bound(), aabb_R.get_max_bound()) + margin
    combo = o3d.geometry.AxisAlignedBoundingBox(minb, maxb)

    ctr = vis.get_view_control()
    ctr.set_lookat(combo.get_center())
    ctr.set_up([0, 0, 1])          # z-up
    ctr.set_front([ -0.5, -0.5, -1 ])  # a reasonable viewing direction

    # Set a zoom that scales with scene extent
    extent = np.linalg.norm(combo.get_extent())
    ctr.set_zoom(0.8 if extent < 20 else 0.9)

    print("[Viz] LiDAR points (shown):", np.asarray(lidar_pcd.points).shape[0])
    print("[Viz] Radar→LiDAR points (shown):", np.asarray(radar_pcd.points).shape[0])

    vis.run()
    vis.destroy_window()

def visualize_xy_overlay(radar_xy, lidar_xyz, T, lidar_subsample=100000, radar_size=6, lidar_size=0.5):
    """
    Quick 2D XY plot (LiDAR XY in gray, radar→LiDAR XY in red).
    """
    radar_xyz = np.c_[np.asarray(radar_xy, float)[:,0], np.asarray(radar_xy, float)[:,1], np.zeros(len(radar_xy))]
    radar_L = apply_T(np.asarray(T, float), radar_xyz)

    L = np.asarray(lidar_xyz, float)
    if L.shape[0] > lidar_subsample:
        idx = np.random.choice(L.shape[0], lidar_subsample, replace=False)
        L = L[idx]

    plt.figure(figsize=(7,7))
    plt.scatter(L[:,0], L[:,1], s=lidar_size, c='#999999', alpha=0.3, label="LiDAR XY")
    plt.scatter(radar_L[:,0], radar_L[:,1], s=radar_size, c='r', alpha=0.9, label="Radar→LiDAR XY")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("XY Overlay (after T_lidar<-radar)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

