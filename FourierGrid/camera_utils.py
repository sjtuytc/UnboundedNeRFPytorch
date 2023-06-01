import numpy as np
import enum
from dataclasses import dataclass
from typing import List, Mapping, Optional, Text, Tuple, Union
from torch import Tensor
import torch


@dataclass
class Rays:
    origins: Tensor
    directions: Tensor
    viewdirs: Tensor
    radii: Tensor
    near: Tensor
    far: Tensor


def split_rays(rays, batch_size):
    ret = []
    origins_all = rays.origins.split(batch_size)
    directions_all = rays.directions.split(batch_size)
    viewdirs_all = rays.viewdirs.split(batch_size)
    radii_all = rays.radii.split(batch_size)
    near_all = rays.near.split(batch_size)
    far_all  = rays.far.split(batch_size)

    for o, d, v, r, n, f in zip(origins_all, directions_all, viewdirs_all, radii_all, near_all, far_all):
        ret.append(Rays(o, d, v, r, n, f))
    return ret

def intrinsic_matrix(fx: float,
                     fy: float,
                     cx: float,
                     cy: float,
                     ):
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.]], dtype=np.float32)


class ProjectionType(enum.Enum):
    """Camera projection type (standard perspective pinhole or fisheye model)."""
    PERSPECTIVE = 'perspective'
    FISHEYE = 'fisheye'


def convert_to_ndc(origins: Tensor,
                   directions: Tensor,
                   pixtocam: Tensor,
                   near: float = 1.) -> Tuple[Tensor, Tensor]:
    """Converts a set of rays to normalized device coordinates (NDC).

    Args:
      origins: ndarray(float32), [..., 3], world space ray origins.
      directions: ndarray(float32), [..., 3], world space ray directions.
      pixtocam: ndarray(float32), [3, 3], inverse intrinsic matrix.
      near: float, near plane along the negative z axis.
      xnp: either numpy or jax.numpy.

    Returns:
      origins_ndc: ndarray(float32), [..., 3].
      directions_ndc: ndarray(float32), [..., 3].

    This function assumes input rays should be mapped into the NDC space for a
    perspective projection pinhole camera, with identity extrinsic matrix (pose)
    and intrinsic parameters defined by inputs focal, width, and height.

    The near value specifies the near plane of the frustum, and the far plane is
    assumed to be infinity.

    The ray bundle for the identity pose camera will be remapped to parallel rays
    within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
    world space can be remapped as long as it has dz < 0 (ray direction has a
    negative z-coord); this allows us to share a common NDC space for "forward
    facing" scenes.

    Note that
        projection(origins + t * directions)
    will NOT be equal to
        origins_ndc + t * directions_ndc
    and that the directions_ndc are not unit length. Rather, directions_ndc is
    defined such that the valid near and far planes in NDC will be 0 and 1.

    See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
    """

    # Shift ray origins to near plane, such that oz = -near.
    # This makes the new near bound equal to 0.
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = torch.moveaxis(directions, -1, 0)
    ox, oy, oz = torch.moveaxis(origins, -1, 0)

    xmult = 1. / pixtocam[0, 2]  # Equal to -2. * focal / cx
    ymult = 1. / pixtocam[1, 2]  # Equal to -2. * focal / cy

    # Perspective projection into NDC for the t = 0 near points
    #     origins + 0 * directions
    origins_ndc = torch.stack([xmult * ox / oz, ymult * oy / oz,
                               -torch.ones_like(oz)], dim=-1)

    # Perspective projection into NDC for the t = infinity far points
    #     origins + infinity * directions
    infinity_ndc = torch.stack([xmult * dx / dz, ymult * dy / dz,
                                torch.ones_like(oz)],
                               dim=-1)

    # directions_ndc points from origins_ndc to infinity_ndc
    directions_ndc = infinity_ndc - origins_ndc

    return origins_ndc, directions_ndc


def pixels_to_rays(
        pix_x_int: Tensor,
        pix_y_int: Tensor,
        pixtocams: Tensor,
        camtoworlds: Tensor,
        distortion_params: Optional[Mapping[str, float]] = None,
        pixtocam_ndc: Optional[Tensor] = None,
        camtype: ProjectionType = ProjectionType.PERSPECTIVE,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Calculates rays given pixel coordinates, intrinisics, and extrinsics.

    Given 2D pixel coordinates pix_x_int, pix_y_int for cameras with
    inverse intrinsics pixtocams and extrinsics camtoworlds (and optional
    distortion coefficients distortion_params and NDC space projection matrix
    pixtocam_ndc), computes the corresponding 3D camera rays.

    Vectorized over the leading dimensions of the first four arguments.

    Args:
      pix_x_int: int array, shape SH, x coordinates of image pixels.
      pix_y_int: int array, shape SH, y coordinates of image pixels.
      pixtocams: float array, broadcastable to SH + [3, 3], inverse intrinsics.
      camtoworlds: float array, broadcastable to SH + [3, 4], camera extrinsics.
      distortion_params: dict of floats, optional camera distortion parameters.
      pixtocam_ndc: float array, [3, 3], optional inverse intrinsics for NDC.
      camtype: camera_utils.ProjectionType, fisheye or perspective camera.
      xnp: either numpy or jax.numpy.

    Returns:
      origins: float array, shape SH + [3], ray origin points.
      directions: float array, shape SH + [3], ray direction vectors.
      viewdirs: float array, shape SH + [3], normalized ray direction vectors.
      radii: float array, shape SH + [1], ray differential radii.
      imageplane: float array, shape SH + [2], xy coordinates on the image plane.
        If the image plane is at world space distance 1 from the pinhole, then
        imageplane will be the xy coordinates of a pixel in that space (so the
        camera ray direction at the origin would be (x, y, -1) in OpenGL coords).
    """

    # Must add half pixel offset to shoot rays through pixel centers.
    def pix_to_dir(x, y):
        return torch.stack([x + .5, y + .5, torch.ones_like(x)], dim=-1)

    # We need the dx and dy rays to calculate ray radii for mip-NeRF cones.
    pixel_dirs_stacked = torch.stack([
        pix_to_dir(pix_x_int, pix_y_int),
        pix_to_dir(pix_x_int + 1, pix_y_int),
        pix_to_dir(pix_x_int, pix_y_int + 1)
    ], dim=0)

    # For jax, need to specify high-precision matmul.
    mat_vec_mul = lambda A, b: torch.matmul(A, b[..., None])[..., 0]

    # Apply inverse intrinsic matrices.
    camera_dirs_stacked = mat_vec_mul(pixtocams, pixel_dirs_stacked)

    if distortion_params is not None:
        # Correct for distortion.
        x, y = _radial_and_tangential_undistort(
            camera_dirs_stacked[..., 0],
            camera_dirs_stacked[..., 1],
            **distortion_params)
        camera_dirs_stacked = torch.stack([x, y, torch.ones_like(x)], -1)

    if camtype == ProjectionType.FISHEYE:
        theta = torch.sqrt(torch.sum(torch.square(camera_dirs_stacked[..., :2]), dim=-1))
        theta = theta.clip(None, np.pi)

        sin_theta_over_theta = torch.sin(theta) / theta
        camera_dirs_stacked = torch.stack([
            camera_dirs_stacked[..., 0] * sin_theta_over_theta,
            camera_dirs_stacked[..., 1] * sin_theta_over_theta,
            torch.cos(theta),
            ], dim=-1)

    # Flip from OpenCV to OpenGL coordinate system.
    camera_dirs_stacked = torch.matmul(camera_dirs_stacked,
                                       torch.diag(torch.tensor([1., -1., -1.],
                                                               dtype=torch.float32,
                                                               device=camera_dirs_stacked.device)))

    # Extract 2D image plane (x, y) coordinates.
    imageplane = camera_dirs_stacked[0, ..., :2]

    # Apply camera rotation matrices.
    directions_stacked = mat_vec_mul(camtoworlds[..., :3, :3],
                                     camera_dirs_stacked)
    # Extract the offset rays.
    directions, dx, dy = directions_stacked

    origins = torch.broadcast_to(camtoworlds[..., :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(directions, ord=2, dim=-1, keepdim=True)

    if pixtocam_ndc is None:
        # Distance from each unit-norm direction vector to its neighbors.
        dx_norm = torch.linalg.norm(dx - directions, ord=2, dim=-1)
        dy_norm = torch.linalg.norm(dy - directions, ord=2, dim=-1)
    else:
        # Convert ray origins and directions into projective NDC space.
        origins_dx, _ = convert_to_ndc(origins, dx, pixtocam_ndc)
        origins_dy, _ = convert_to_ndc(origins, dy, pixtocam_ndc)
        origins, directions = convert_to_ndc(origins, directions, pixtocam_ndc)

        # In NDC space, we use the offset between origins instead of directions.
        dx_norm = torch.linalg.norm(origins_dx - origins, ord=2, dim=-1)
        dy_norm = torch.linalg.norm(origins_dy - origins, ord=2, dim=-1)

    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see the original mipnerf paper).
    radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / np.sqrt(12.)

    return origins, directions, viewdirs, radii, imageplane


def _compute_residual_and_jacobian(
        x: Tensor,
        y: Tensor,
        xd: Tensor,
        yd: Tensor,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        k4: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Auxiliary function of radial_and_tangential_undistort()."""
    # Adapted from https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = (k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4)))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(
        xd: Tensor,
        yd: Tensor,
        k1: float = 0,
        k2: float = 0,
        k3: float = 0,
        k4: float = 0,
        p1: float = 0,
        p2: float = 0,
        eps: float = 1e-9,
        max_iterations=10) -> Tuple[Tensor, Tensor]:
    """Computes undistorted (x, y) from (xd, yd)."""
    # From https://github.com/google/nerfies/blob/main/nerfies/camera.py
    # Initialize from the distorted point.
    x = xd.clone()
    y = yd.clone()

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=xd, yd=yd, k1=k1, k2=k2, k3=k3, k4=k4, p1=p1, p2=p2)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps, x_numerator / denominator,
            torch.zeros_like(denominator))
        step_y = torch.where(
            torch.abs(denominator) > eps, y_numerator / denominator,
            torch.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return x, y
