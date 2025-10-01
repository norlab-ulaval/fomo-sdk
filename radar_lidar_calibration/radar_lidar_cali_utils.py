
import numpy as np
from scipy import ndimage


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


# icp utils
import open3d as o3d
import matplotlib.pyplot as plt

def apply_T(T, P):
    P = np.asarray(P, dtype=np.float64)
    return (P @ T[:3, :3].T) + T[:3, 3]

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


def icp_multistage(radar_pts, lidar_pts, T_init=None, crop_margin=5.0, verbose=True):
    """
    Radar->LiDAR ICP constrained to x, y, yaw. z/roll/pitch are frozen to T_init.
    Designed to accept small (~2 cm) corrections when they truly improve fit.
    """

    # -------- helpers --------
    def apply_T(T, P):
        P = np.asarray(P, float)
        return (P @ T[:3,:3].T) + T[:3,3]

    def to_pcd(P, voxel=None):
        P = np.asarray(P, float)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        if voxel and voxel > 0.0:
            pcd = pcd.voxel_down_sample(voxel)
        return pcd  # p2p only -> no normals needed

    def rpy_from_R_xyz(R):
        pitch = -np.arcsin(np.clip(R[2,0], -1, 1))
        roll  = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(R[1,0], R[0,0])
        return roll, pitch, yaw

    def yaw_from_R(R):
        return np.arctan2(R[1,0], R[0,0])

    def project_xy_yaw(T_cur, T_prior):
        """Keep x,y,yaw from T_cur; keep z, roll, pitch from T_prior."""
        r0, p0, _ = rpy_from_R_xyz(T_prior[:3,:3])
        _, _, y   = rpy_from_R_xyz(T_cur[:3,:3])
        cy, sy = np.cos(y), np.sin(y)
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        Rx = lambda a: np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        Ry = lambda a: np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        R_new = Rz @ Ry(p0) @ Rx(r0)
        t_new = T_cur[:3,3].copy(); t_new[2] = T_prior[2,3]
        Tout = np.eye(4); Tout[:3,:3] = R_new; Tout[:3,3] = t_new
        return Tout

    def clamp_xy_yaw_update(Delta, max_step_xy, max_step_yaw_deg):
        """Clamp incremental SE(3) 'Delta' to bounded SE(2) (dx,dy,dyaw)."""
        dx, dy = float(Delta[0,3]), float(Delta[1,3])
        dyaw = yaw_from_R(Delta[:3,:3])
        max_yaw = np.deg2rad(max_step_yaw_deg)
        dyaw = np.clip(dyaw, -max_yaw, max_yaw)
        step = np.hypot(dx, dy)
        if step > max_step_xy and step > 1e-12:
            s = max_step_xy / step
            dx *= s; dy *= s
        c, s = np.cos(dyaw), np.sin(dyaw)
        D = np.eye(4)
        D[:3,:3] = np.array([[ c,-s,0],[ s, c,0],[0,0,1]])
        D[0,3] = dx; D[1,3] = dy
        return D

    def shrink_toward_identity(Delta, shrink=0.8):
        """Scale the incremental update toward identity (0..1)."""
        dx, dy = float(Delta[0,3]), float(Delta[1,3])
        dyaw = yaw_from_R(Delta[:3,:3])
        dx *= shrink; dy *= shrink; dyaw *= shrink
        c, s = np.cos(dyaw), np.sin(dyaw)
        D = np.eye(4)
        D[:3,:3] = np.array([[ c,-s,0],[ s, c,0],[0,0,1]])
        D[0,3] = dx; D[1,3] = dy
        return D

    def bound_total_xy_yaw(T, T_prior, max_xy=0.05, max_yaw_deg=0.5):
        """Clip final deviation from prior."""
        yaw0 = yaw_from_R(T_prior[:3,:3]); yaw1 = yaw_from_R(T[:3,:3])
        dpsi = np.arctan2(np.sin(yaw1-yaw0), np.cos(yaw1-yaw0))
        dpsi = np.clip(dpsi, -np.deg2rad(max_yaw_deg), np.deg2rad(max_yaw_deg))
        c,s = np.cos(yaw0+dpsi), np.sin(yaw0+dpsi)
        Rz = np.array([[ c,-s,0],[ s, c,0],[0,0,1]])
        dx, dy = T[0,3]-T_prior[0,3], T[1,3]-T_prior[1,3]
        r = np.hypot(dx,dy)
        if r > max_xy and r > 1e-12:
            sc = max_xy/r
            dx *= sc; dy *= sc
        r0,p0,_ = rpy_from_R_xyz(T_prior[:3,:3])
        Rx = lambda a: np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
        Ry = lambda a: np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
        Tout = np.eye(4)
        Tout[:3,:3] = Rz @ Ry(p0) @ Rx(r0)
        Tout[:3,3]  = T_prior[:3,3] + np.array([dx,dy,0.0])
        return Tout

    def eval_rmse_fit(radar_pcd, lidar_pcd, T, thresh=0.10):  # tighter radius
        r = to_pcd(apply_T(T, np.asarray(radar_pcd.points)), voxel=0.03)
        l = to_pcd(np.asarray(lidar_pcd.points),             voxel=0.03)
        ev = o3d.pipelines.registration.evaluate_registration(r, l, thresh)
        return float(ev.inlier_rmse), float(ev.fitness)

    # -------- inputs --------
    radar_pts = np.asarray(radar_pts, float)
    if radar_pts.ndim != 2 or radar_pts.shape[1] < 2:
        raise ValueError("radar_pts must be (N,2) or (N,>=2).")
    radar_xyz = np.c_[radar_pts[:,0], radar_pts[:,1], np.zeros(len(radar_pts))]
    lidar_xyz = np.asarray(lidar_pts, float)
    if lidar_xyz.ndim != 2 or lidar_xyz.shape[1] != 3:
        raise ValueError("lidar_pts must be (M,3).")
    radar_xyz = radar_xyz[np.isfinite(radar_xyz).all(axis=1)]
    lidar_xyz = lidar_xyz[np.isfinite(lidar_xyz).all(axis=1)]

    T = np.eye(4, dtype=float) if T_init is None else np.array(T_init, float)
    T_prior = T.copy()

    # XY crop around transformed radar 
    if crop_margin and crop_margin > 0:
        r0 = apply_T(T, radar_xyz)
        rx_min, ry_min = r0[:,0].min()-crop_margin, r0[:,1].min()-crop_margin
        rx_max, ry_max = r0[:,0].max()+crop_margin, r0[:,1].max()+crop_margin
        m = (lidar_xyz[:,0]>=rx_min)&(lidar_xyz[:,0]<=rx_max)&(lidar_xyz[:,1]>=ry_min)&(lidar_xyz[:,1]<=ry_max)
        cropped = lidar_xyz[m]
        if len(cropped) >= 1000:
            lidar_xyz = cropped
            if verbose: print(f"[ICP] XY-crop kept {len(lidar_xyz)} LiDAR points.")
        elif verbose:
            print("[ICP] Skipping crop (too few points kept).")

    # base clouds 
    radar_pcd = to_pcd(radar_xyz, voxel=0.10)
    lidar_pcd = to_pcd(lidar_xyz, voxel=0.10)
    # Estimators: plain p2p and robust p2p (Tukey) for finer stages
    def make_pt2pt_estimator(tukey_k=0.3):
        reg = o3d.pipelines.registration
        try:
            rk = reg.RobustKernel(reg.RobustKernelType.Tukey, tukey_k)
            return reg.TransformationEstimationPointToPoint(robust_kernel=rk)
        except Exception:
            try:
                rk = reg.RobustKernel(method="tukey", scaling=tukey_k)
                return reg.TransformationEstimationPointToPoint(robust_kernel=rk)
            except Exception:
                return reg.TransformationEstimationPointToPoint()

    est_p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    est_p2p_robust = make_pt2pt_estimator(tukey_k=0.3)

    stages = [
        dict(voxel=0.25, max_corr_dist=1.50, iters=60, estimator=est_p2p),
        dict(voxel=0.15, max_corr_dist=0.80, iters=60, estimator=est_p2p),
        dict(voxel=0.10, max_corr_dist=0.50, iters=80, estimator=est_p2p_robust),
        dict(voxel=0.05, max_corr_dist=0.25, iters=100, estimator=est_p2p_robust),
        dict(voxel=0.03, max_corr_dist=0.15, iters=120, estimator=est_p2p_robust),
    ]
    max_step_xy      = [0.10, 0.05, 0.03, 0.02, 0.01]   # m
    max_step_yaw_deg = [1.00, 0.50, 0.30, 0.20, 0.10]   # deg
    shrink_factor    = [0.8,  0.7,  0.6,  0.5,  0.5]    # gentler shrink

    # Evaluate CAD first (tighter threshold so small changes matter)
    rmse_cad, fit_cad = eval_rmse_fit(radar_pcd, lidar_pcd, T_prior, thresh=0.10)

    for k, s in enumerate(stages):
        radar_stage = to_pcd(apply_T(T, np.asarray(radar_pcd.points)), voxel=s["voxel"])
        lidar_stage = to_pcd(np.asarray(lidar_pcd.points),             voxel=s["voxel"])

        reg = o3d.pipelines.registration.registration_icp(
            radar_stage, lidar_stage,
            s["max_corr_dist"], np.eye(4), s["estimator"],
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=s["iters"])
        )

        Delta = reg.transformation
        Delta = clamp_xy_yaw_update(Delta, max_step_xy[k], max_step_yaw_deg[k])
        Delta = shrink_toward_identity(Delta, shrink=shrink_factor[k])
        T = project_xy_yaw(Delta @ T, T_prior)

        if verbose:
            dx, dy = T[0,3]-T_prior[0,3], T[1,3]-T_prior[1,3]
            dpsi = np.rad2deg(np.arctan2(np.sin(yaw_from_R(T[:3,:3]) - yaw_from_R(T_prior[:3,:3])),
                                         np.cos(yaw_from_R(T[:3,:3]) - yaw_from_R(T_prior[:3,:3]))))
            print(f"[ICP] voxel={s['voxel']:.2f} corr={s['max_corr_dist']:.2f} fit={reg.fitness:.3f} rmse={reg.inlier_rmse:.3f} | Δxy={np.hypot(dx,dy):.3f} m, Δyaw={dpsi:.3f}°")

    # acceptance test 
    rmse_new, fit_new = eval_rmse_fit(radar_pcd, lidar_pcd, T, thresh=0.10)
    improv = (rmse_cad - rmse_new) / max(rmse_cad, 1e-9)

    ACCEPT_MIN_IMPROV = 0.02   # 2% better than CAD
    ACCEPT_MIN_MOVE   = 0.005  # at least 5 mm move OR 0.05°
    ACCEPT_MIN_YAW    = np.deg2rad(0.05)

    dx_tot, dy_tot = T[0,3]-T_prior[0,3], T[1,3]-T_prior[1,3]
    dpsi_tot = np.arctan2(np.sin(yaw_from_R(T[:3,:3]) - yaw_from_R(T_prior[:3,:3])),
                          np.cos(yaw_from_R(T[:3,:3]) - yaw_from_R(T_prior[:3,:3])))

    moved_enough = (np.hypot(dx_tot, dy_tot) >= ACCEPT_MIN_MOVE) or (abs(dpsi_tot) >= ACCEPT_MIN_YAW)
    if improv < ACCEPT_MIN_IMPROV and not moved_enough:
        if verbose:
            print(f"[ICP] Rejecting update: improvement {improv*100:.1f}% < {ACCEPT_MIN_IMPROV*100:.0f}% and move too small. Keeping CAD.")
        T = T_prior
        rmse_new, fit_new = rmse_cad, fit_cad
    else:
        if verbose:
            print(f"[ICP] Accepting update: improvement {improv*100:.1f}% (CAD RMSE {rmse_cad:.4f} -> {rmse_new:.4f})")

    # final hard bound toward CAD 
    T = bound_total_xy_yaw(T, T_prior, max_xy=0.03, max_yaw_deg=0.5)

    if verbose:
        print("\nICP (x,y,yaw) Result - T_ref:\n", T)
        print("Fitness:", fit_new, "RMSE:", rmse_new)

    return T, float(fit_new), float(rmse_new)



def se2_from_T(T):
    """Extract (x,y,yaw) from SE(3) assuming z/roll/pitch negligible."""
    x, y = float(T[0,3]), float(T[1,3])
    yaw = np.arctan2(T[1,0], T[0,0])
    return np.array([x, y, yaw], dtype=float)

def T_from_se2_xyyaw(xyyaw, T_prior):
    """Compose SE(3) from (x,y,yaw) and prior z/roll/pitch."""
    x, y, yaw = float(xyyaw[0]), float(xyyaw[1]), float(xyyaw[2])
    r0, p0, _ = rpy_from_R_xyz(T_prior[:3,:3])
    R = Rz(yaw) @ Ry(p0) @ Rx(r0)
    Tout = np.eye(4)
    Tout[:3,:3] = R
    Tout[:3,3]  = np.array([x, y, T_prior[2,3]])
    return Tout

def circular_mean(angles):
    s = np.sin(angles).mean()
    c = np.cos(angles).mean()
    return np.arctan2(s, c)

def se2_median_robust(T_list, T_prior, max_iters=5):
    """Robust median-like aggregation over (x,y,yaw), guarding against outliers."""
    if len(T_list) == 0:
        return np.array(T_prior, float)
    X = np.stack([se2_from_T(T) for T in T_list], axis=0)
    x0 = np.median(X[:,0])
    y0 = np.median(X[:,1])
    yaw0 = circular_mean(X[:,2])
    m = np.array([x0, y0, yaw0])
    for _ in range(max_iters):
        dxy = np.hypot(X[:,0]-m[0], X[:,1]-m[1])
        dyaw = np.arctan2(np.sin(X[:,2]-m[2]), np.cos(X[:,2]-m[2]))
        d = np.sqrt(dxy**2 + (0.5*dyaw)**2)
        w = 1.0 / np.clip(d, 1e-6, None)
        w /= w.sum()
        x = np.sum(w * X[:,0])
        y = np.sum(w * X[:,1])
        s = np.sum(w * np.sin(X[:,2]))
        c = np.sum(w * np.cos(X[:,2]))
        yaw = np.arctan2(s, c)
        new_m = np.array([x, y, yaw])
        if np.linalg.norm(new_m - m) < 1e-6:
            m = new_m
            break
        m = new_m
    return T_from_se2_xyyaw(m, T_prior)

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

def visualize_xy_overlay(radar_xy, lidar_xyz, T, lidar_subsample=100000, radar_size=5, lidar_size=4.0):
    """
    Quick 2D XY plot (LiDAR XY in bright blue, radar→LiDAR XY in red).
    """
    radar_xyz = np.c_[np.asarray(radar_xy, float)[:,0], np.asarray(radar_xy, float)[:,1], np.zeros(len(radar_xy))]
    radar_L = apply_T(np.asarray(T, float), radar_xyz)

    L = np.asarray(lidar_xyz, float)
    if L.shape[0] > lidar_subsample:
        idx = np.random.choice(L.shape[0], lidar_subsample, replace=False)
        L = L[idx]

    plt.figure(figsize=(7,7))
    plt.scatter(L[:,0], L[:,1], s=lidar_size, c='#0066FF', alpha=0.7, label="LiDAR XY", edgecolors='#003399', linewidth=0.3)
    plt.scatter(radar_L[:,0], radar_L[:,1], s=radar_size, c='r', alpha=0.9, label="Radar→LiDAR XY", edgecolors='darkred', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("XY Overlay (after T_lidar<-radar)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def crop_lidar_by_height(lidar_xyz, radar_height_m, lidar_height_m, tol=0.25):
    """
    Keep LiDAR points whose z is within ±tol of the radar plane height,
    assuming z-up and level rig.

    lidar_xyz: (M,3) points in LiDAR frame [x,y,z]
    radar_height_m: radar height above ground (m), e.g., 1.235
    lidar_height_m: lidar height above ground (m), e.g., 1.114
    tol: half-thickness around the plane, in meters

    Returns: cropped_points, mask (boolean)
    """
    lidar_xyz = np.asarray(lidar_xyz, float)
    z_radar_in_lidar = float(radar_height_m - lidar_height_m)  # e.g., 0.121 m
    z = lidar_xyz[:, 2]
    mask = np.abs(z - z_radar_in_lidar) <= float(tol)
    return lidar_xyz[mask]

def crop_lidar_by_range(lidar_xyz, r_max, r_min=0.0, use_xy=False,
                        origin=None, return_mask=False):
    """
    Keep LiDAR points within [r_min, r_max] of 'origin'.

    Args
    ----
    lidar_xyz : (N,3) float array  (points in LiDAR frame)
    r_max     : float, outer radius in meters
    r_min     : float, inner radius in meters (default 0)
    use_xy    : bool, if True use horizontal (XY) range; else full 3D range
    origin    : None or (3,) array, center of the ring (default [0,0,0])
    return_mask : bool, if True also return the boolean mask

    Returns
    -------
    cropped : (M,3) array of points inside the range (and mask if requested)
    """
    pts = np.asarray(lidar_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("lidar_xyz must be (N,3).")
    if r_max <= 0 or r_min < 0 or r_min > r_max:
        raise ValueError("Require 0 <= r_min <= r_max and r_max > 0.")

    o = np.zeros(3) if origin is None else np.asarray(origin, float).reshape(3)

    # Use squared distances (avoids sqrt for speed)
    if use_xy:
        v = pts[:, :2] - o[:2]
        d2 = v[:, 0]**2 + v[:, 1]**2
        rmin2, rmax2 = r_min**2, r_max**2
    else:
        v = pts - o
        d2 = (v * v).sum(axis=1)
        rmin2, rmax2 = r_min**2, r_max**2

    finite = np.isfinite(pts).all(axis=1)
    mask = finite & (d2 >= rmin2) & (d2 <= rmax2)
    cropped = pts[mask]

    return (cropped, mask) if return_mask else cropped

# do a check here
def invert_se3(T):
    R, t = T[:3,:3], T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti
