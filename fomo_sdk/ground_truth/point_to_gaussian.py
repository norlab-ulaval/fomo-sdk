import os

import numpy as np
import pandas as pd

# import jax
# import jax.numpy as jnp
import tqdm
from matplotlib import pyplot as plt
from pypointmatcher import pointmatcher as pm
from pypointmatcher import pointmatchersupport as pms
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from utils import point_to_point_minimization

PM = pm.PointMatcher
DP = PM.DataPoints
Parameters = pms.Parametrizable.Parameters

TF_FILE = "gnss_tf_final.csv"
ICP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "point_to_gaussian.yaml")

errors_history = []
params_history = []

D1 = 1
D2 = 0.001


def draw_ellipsoid(ax, mean, cov, color="g", alpha=0.2, n_std=1):
    """
    Draw a 3D ellipsoid representing the covariance matrix.

    Parameters:
    - ax: 3D matplotlib axis
    - mean: center of the ellipsoid (3D point)
    - cov: 3x3 covariance matrix
    - color: color of the ellipsoid
    - alpha: transparency
    - n_std: number of standard deviations for the ellipsoid size
    """
    # Eigenvalue decomposition
    eig_vals, eig_vecs = np.linalg.eig(cov[:3, :3])  # Use only 3x3 part for 3D

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Semi-axes lengths (scaled by number of standard deviations)
    radii = n_std * np.sqrt(eig_vals)

    # Generate sphere points
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale by radii
    x_ellipsoid = radii[0] * x_sphere
    y_ellipsoid = radii[1] * y_sphere
    z_ellipsoid = radii[2] * z_sphere

    # Rotate using eigenvectors
    for i in range(len(x_sphere)):
        for j in range(len(x_sphere)):
            point = np.array([x_ellipsoid[i, j], y_ellipsoid[i, j], z_ellipsoid[i, j]])
            rotated_point = eig_vecs @ point
            x_ellipsoid[i, j] = rotated_point[0]
            y_ellipsoid[i, j] = rotated_point[1]
            z_ellipsoid[i, j] = rotated_point[2]

    # Translate to mean position
    x_ellipsoid += mean[0]
    y_ellipsoid += mean[1]
    z_ellipsoid += mean[2]

    # Plot the ellipsoid
    ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, color=color, alpha=alpha)


def plot_gnss_geom(
    ax, geom, T_geom, label="Transformed gnss geometry", color="red", alpha=0.5
):
    transformed_gnss_geometry = T_geom @ np.vstack((geom, np.ones(3)))
    ax.scatter(
        transformed_gnss_geometry[0],
        transformed_gnss_geometry[1],
        transformed_gnss_geometry[2],
        c=color,
        label=label,
    )
    ax.text(
        transformed_gnss_geometry[0][0],
        transformed_gnss_geometry[1][0],
        transformed_gnss_geometry[2][0],
        "1",
        color=color,
        fontsize=12,
    )
    ax.text(
        transformed_gnss_geometry[0][1],
        transformed_gnss_geometry[1][1],
        transformed_gnss_geometry[2][1],
        "2",
        color=color,
        fontsize=12,
    )
    ax.text(
        transformed_gnss_geometry[0][2],
        transformed_gnss_geometry[1][2],
        transformed_gnss_geometry[2][2],
        "3",
        color=color,
        fontsize=12,
    )
    # connect the points with lines
    ax.plot(
        transformed_gnss_geometry[0],
        transformed_gnss_geometry[1],
        transformed_gnss_geometry[2],
        c=color,
    )
    # connect also the last and the first points
    ax.plot(
        [transformed_gnss_geometry[0][0], transformed_gnss_geometry[0][-1]],
        [transformed_gnss_geometry[1][0], transformed_gnss_geometry[1][-1]],
        [transformed_gnss_geometry[2][0], transformed_gnss_geometry[2][-1]],
        c=color,
        alpha=0.5,
    )


def transform(p, x, rotation_scale=1):
    tx, ty, tz, rx, ry, rz = p
    rx *= rotation_scale
    ry *= rotation_scale
    rz *= rotation_scale
    # rotation matrices
    Rx = jnp.array(
        [[1, 0, 0], [0, jnp.cos(rx), -jnp.sin(rx)], [0, jnp.sin(rx), jnp.cos(rx)]]
    )
    Ry = jnp.array(
        [[jnp.cos(ry), 0, jnp.sin(ry)], [0, 1, 0], [-jnp.sin(ry), 0, jnp.cos(ry)]]
    )
    Rz = jnp.array(
        [[jnp.cos(rz), -jnp.sin(rz), 0], [jnp.sin(rz), jnp.cos(rz), 0], [0, 0, 1]]
    )
    R = Rx @ Ry @ Rz
    return R @ x + jnp.array([tx, ty, tz])


def objective(params, points, means, covs):
    # input(f"Points: {points}, Means: {means}" )
    score = 0
    for k in range(points.shape[1]):
        x_k = points[0:3, k]
        mu_k = means[0:3, k]
        y_k = transform(params, x_k)
        r = y_k - mu_k

        cov = covs[k, 0:3, 0:3]
        inv = np.linalg.inv(cov)
        score += 0.5 * r.T @ inv @ r
        # score += -D1 * jnp.exp(-0.5*D2 * r.T @ inv @ r)
        # score += 0.5 * r.T @ r
    # print(score.item())
    return score


# gradient_obj = jax.grad(objective)
# hessian_obj = jax.hessian(objective)


def create_transform_matrix_from_trans_euler(params) -> np.ndarray:
    """
    Args:
        params (np.ndarray): The parameters to optimize. A 6x1 vector containing xyz and the xyz euler angles.

    Returns:
        np.ndarray: The 4x4 transformation matrix.
    """
    # Extract translation and rotation
    translation = params[:3]
    euler = params[3:]
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("zyx", euler).as_matrix()
    T[:3, 3] = translation
    return T


def point_to_gaussian_minimization(p, mu, cov):
    initial_guess = np.zeros(6)
    bounds = [
        (-0.1, 0.1),
        (-0.1, 0.1),
        (-0.1, 0.1),  # translation bounds (meters)
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.5),
    ]  # rotation bounds (radians)

    result = minimize(
        fun=objective,
        x0=initial_guess,
        jac=gradient_obj,
        hess=hessian_obj,
        args=(p, mu, cov),
        method="trust-exact",
        bounds=bounds,
        # method='L-BFGS-B',
        options={"maxiter": 100, "ftol": 1e-9, "gtol": 1e-9},
    )

    return create_transform_matrix_from_trans_euler(result.x)


def point_to_gaussian_minimization_lpm(p, mu, cov, icp):
    p = np.vstack([p, np.ones((1, 3))])

    featLabels = DP.Labels()
    featLabels.append(DP.Label("x", 1))
    featLabels.append(DP.Label("y", 1))
    featLabels.append(DP.Label("z", 1))
    featLabels.append(DP.Label("pad", 1))

    descLabels = DP.Labels()
    descLabels.append(DP.Label("eigValues", 3))
    descLabels.append(DP.Label("eigVectors", 9))
    point_descriptors = np.zeros((12, 3))
    for i in range(mu.shape[1]):
        cov_i = cov[i, :, :]
        eigValues, eigVectors = np.linalg.eig(cov_i)
        point_descriptors[:, i] = np.hstack([eigValues.flatten(), eigVectors.flatten()])

    dp_reading = DP(p, featLabels)
    dp_reference = DP(mu, featLabels, point_descriptors, descLabels)

    T_icp = icp(dp_reading, dp_reference)

    return T_icp


def point_to_gaussian_df(
    gnss_geometry,
    means_df: pd.DataFrame,
    covs_df: pd.DataFrame,
    visualize=True,
    use_lpm=True,
    add_random_noise=False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    traj = []
    covs = []
    for idx in tqdm.tqdm(range(len(means_df)), desc="P2G Processing trajectory"):
        row_mu = means_df.iloc[idx, 1:].to_numpy().reshape(3, 3).T
        row_cov = covs_df.iloc[idx, 1:].to_numpy().reshape(3, 3, 3)

        # initial transform to speed up convergence
        t, R = point_to_point_minimization(gnss_geometry, row_mu)
        if add_random_noise:
            t = t + np.random.normal(0, 0.1, (3,))
            R_2 = Rotation.from_euler("zyx", [0.009, 0.01, 0.004]).as_matrix()
            R = R_2 @ R
        T_init = np.eye(4)
        T_init[:3, :3] = R
        T_init[:3, 3] = t

        row_mu_tf = np.vstack([row_mu, np.ones((1, row_mu.shape[1]))])
        T_init_inv = np.linalg.inv(T_init)
        row_mu_tf = T_init_inv @ row_mu_tf

        if use_lpm:
            icp = PM.ICP()
            icp.setDefault()
            icp.loadFromYaml(ICP_CONFIG_PATH)
            try:
                T_optim = point_to_gaussian_minimization_lpm(
                    gnss_geometry, row_mu_tf, row_cov, icp
                )
            except Exception as e:
                print(f"Error in point_to_gaussian_minimization_lpm: {e}")
                print("Skipping to the next frame")
                continue
        else:
            T_optim = point_to_gaussian_minimization(gnss_geometry, row_mu_tf, row_cov)
        T = T_init @ T_optim
        t = T[:3, 3]
        # quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        quat = [0, 0, 0, 1]
        traj.append([means_df.iloc[idx, 0], *t, *quat])

        # compute the result covariance matrix
        W_avg = np.zeros((3, 3))
        Ws = []
        for i in range(3):
            W = np.linalg.inv(row_cov[i])
            Ws.append(W)
            W_avg += W

        covs.append([means_df.iloc[idx, 0], *np.linalg.inv(W_avg).flatten()])

        if not visualize:
            continue
        means = row_mu
        # plot the results in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(means[0], means[1], means[2], c="r", label="Means", alpha=0.5)

        # plot number 1 at the position of the mean in the first column
        ax.text(means[0][0], means[1][0], means[2][0], "1", color="black", fontsize=12)
        ax.text(means[0][1], means[1][1], means[2][1], "2", color="black", fontsize=12)
        ax.text(means[0][2], means[1][2], means[2][2], "3", color="black", fontsize=12)
        # plot covariance as ellipsis around means
        for i in range(means.shape[1]):
            draw_ellipsoid(ax, means[:, i], row_cov[i], color="g", alpha=0.2)

        plot_gnss_geom(ax, gnss_geometry, T, color="black")
        plot_gnss_geom(
            ax, gnss_geometry, T_init, "Initial gnss geometry", color="magenta"
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    df_traj = pd.DataFrame(
        traj, columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    )
    df_covs = pd.DataFrame(
        covs,
        columns=[
            "timestamp",
            "cov_xx",
            "cov_xy",
            "cov_xz",
            "cov_yx",
            "cov_yy",
            "cov_yz",
            "cov_zx",
            "cov_zy",
            "cov_zz",
        ],
    )
    return df_traj, df_covs


def test_point_to_gaussian():
    np.random.seed(42)
    number_of_measurements = 10

    gnss_geometry = np.array(read_tf_file(TF_FILE, 3))
    gnss_geometry[2, :] = np.array([0, 0, 0])
    shift = np.array([[2, 2, 2], [1, 1, 1], [0, 0, 0]])

    # Test case 1: Simple 3D case with single point
    means = gnss_geometry + shift
    means = np.tile(means, (1, number_of_measurements))  # 3x9
    some_noise = np.random.normal(0, 0.05, means.shape)
    means = means + some_noise

    covs = np.array(
        [0.29 * np.eye(4), 0.25 * np.eye(4), 0.35 * np.eye(4)]
    )  # 1x3x3 identity matrix
    covs = np.tile(covs, (number_of_measurements, 1, 1))  # 3x3x3 identity matrix

    gnss_geometry = np.vstack((gnss_geometry, np.ones((1, 3))))
    means = np.vstack((means, np.ones((1, means.shape[1]))))

    plt.figure()
    plt.scatter(means[0], means[1], c="r", label="Means")
    plt.scatter(gnss_geometry[0], gnss_geometry[1], c="b", label="Gnss geometry")

    # plot covariance as ellipsis around means
    for i in range(means.shape[1]):
        mean = means[:, i]
        cov = covs[i]
        # Note: This needs to be updated to use 3D plotting context
        # For now, this 2D case might need special handling
        pass  # Placeholder - consider creating a 2D version or converting to 3D

    # print(point_to_gaussian(gnss_geometry, means, covs))

    res = register_points_to_gaussian(gnss_geometry, means, covs)
    print(res)
    T = create_transform_matrix_from_trans_quat(res.x)
    print(T)

    transformed_gnss_geometry = T @ gnss_geometry
    plt.scatter(
        transformed_gnss_geometry[0],
        transformed_gnss_geometry[1],
        c="y",
        label="Transformed gnss geometry",
    )
    plt.legend()

    plt.figure(2)
    plt.plot(errors_history, label="Means")

    plt.show()


if __name__ == "__main__":
    test_point_to_gaussian()
