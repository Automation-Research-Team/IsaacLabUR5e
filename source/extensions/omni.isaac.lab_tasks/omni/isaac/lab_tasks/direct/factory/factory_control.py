import math
import torch

import omni.isaac.core.utils.torch as torch_utils

from omni.isaac.lab.utils.math import axis_angle_from_quat

def _apply_task_space_gains(
    delta_fingertip_pose, fingertip_midpoint_linvel, fingertip_midpoint_angvel, task_prop_gains, task_deriv_gains
):
    """Interpret PD gains as task-space gains. Apply to task-space error."""

    task_wrench = torch.zeros_like(delta_fingertip_pose)

    # Apply gains to lin error components
    lin_error = delta_fingertip_pose[:, 0:3]
    task_wrench[:, 0:3] = task_prop_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
        0.0 - fingertip_midpoint_linvel
    )

    # Apply gains to rot error components
    rot_error = delta_fingertip_pose[:, 3:6]
    task_wrench[:, 3:6] = task_prop_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
        0.0 - fingertip_midpoint_angvel
    )
    return task_wrench


def get_pose_error(
    fingertip_midpoint_pos,
    fingertip_midpoint_quat,
    ctrl_target_fingertip_midpoint_pos,
    ctrl_target_fingertip_midpoint_quat,
    jacobian_type,
    rot_error_type,
):
    
    pos_error = ctrl_target_fingertip_midpoint_pos - fingertip_midpoint_pos
    
    if jacobian_type == "geometric":  # See example 2.9.8; note use of J_g and transformation between rotation vectors
        
        quat_dot = (ctrl_target_fingertip_midpoint_quat * fingertip_midpoint_quat).sum(dim=1, keepdim=True)
        ctrl_target_fingertip_midpoint_quat = torch.where(
            quat_dot.expand(-1, 4) >= 0, ctrl_target_fingertip_midpoint_quat, -ctrl_target_fingertip_midpoint_quat
        )

        fingertip_midpoint_quat_norm = torch_utils.quat_mul(
            fingertip_midpoint_quat, torch_utils.quat_conjugate(fingertip_midpoint_quat)
        )[
            :, 0
        ]  # scalar component
        fingertip_midpoint_quat_inv = torch_utils.quat_conjugate(
            fingertip_midpoint_quat
        ) / fingertip_midpoint_quat_norm.unsqueeze(-1)
        quat_error = torch_utils.quat_mul(ctrl_target_fingertip_midpoint_quat, fingertip_midpoint_quat_inv)

        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)

    elif (
        jacobian_type == "analytic"
    ):  # See example 2.9.7; note use of J_a and difference of rotation vectors
        # Compute axis-angle error
        axis_angle_error = axis_angle_from_quat(
            ctrl_target_fingertip_midpoint_quat
        ) - axis_angle_from_quat(fingertip_midpoint_quat)


    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        return pos_error, axis_angle_error


def get_delta_dof_pos(delta_pose, ik_method, jacobian, device):
    
    if ik_method == "pinv":  # Jacobian pseudoinverse
        k_val = 1.0
        jacobian_pinv = torch.linalg.pinv(jacobian)
        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "trans":  # Jacobian transpose
        k_val = 1.0
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "dls":  # damped least squares (Levenberg-Marquardt)
        lambda_val = 0.1  # 0.1
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=device)
        delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    elif ik_method == "svd":  # adaptive SVD
        k_val = 1.0
        U, S, Vh = torch.linalg.svd(jacobian)
        S_inv = 1.0 / S
        min_singular_value = 1.0e-5
        S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
        jacobian_pinv = (
            torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6] @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
        )

        delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
        delta_dof_pos = delta_dof_pos.squeeze(-1)

    return delta_dof_pos


def get_analytic_jacobian(fingertip_quat, fingertip_jacobian, num_envs, device):
    
    batch = num_envs
    # Compute E_inv_top (i.e., [E_p_inv, 0])
    I = torch.eye(3, device=device)
    E_p_inv = I.repeat((batch, 1)).reshape(batch, 3, 3)
    E_inv_top = torch.cat((E_p_inv, torch.zeros((batch, 3, 3), device=device)), dim=2)

    # Compute E_inv_bottom (i.e., [0, E_r_inv])
    fingertip_axis_angle = axis_angle_from_quat(fingertip_quat)
    fingertip_axis_angle_cross = get_skew_symm_matrix(
        fingertip_axis_angle, device=device
    )
    fingertip_angle = torch.linalg.vector_norm(fingertip_axis_angle, dim=1)
    factor_1 = 1 / (fingertip_angle**2)
    factor_2 = 1 - fingertip_angle * 0.5 * torch.sin(fingertip_angle) / (
        1 - torch.cos(fingertip_angle)
    )
    factor_3 = factor_1 * factor_2
    E_r_inv = (
        I
        - 1 * 0.5 * fingertip_axis_angle_cross
        + (fingertip_axis_angle_cross @ fingertip_axis_angle_cross)
        * factor_3.unsqueeze(-1).repeat((1, 3 * 3)).reshape((batch, 3, 3))
    )
    E_inv_bottom = torch.cat(
        (torch.zeros((batch, 3, 3), device=device), E_r_inv), dim=2
    )

    E_inv = torch.cat(
        (E_inv_top.reshape((batch, 3 * 6)), E_inv_bottom.reshape((batch, 3 * 6))), dim=1
    ).reshape((batch, 6, 6))

    J_a = E_inv @ fingertip_jacobian

    return J_a


def get_skew_symm_matrix(vec, device):
    """Convert vector to skew-symmetric matrix."""
    # Reference: https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication

    batch = vec.shape[0]
    I = torch.eye(3, device=device)
    skew_symm = torch.transpose(
        torch.cross(
            vec.repeat((1, 3)).reshape((batch * 3, 3)), I.repeat((batch, 1))
        ).reshape(batch, 3, 3),
        dim0=1,
        dim1=2,
    )

    return skew_symm
