# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR

from .factory_tasks_cfg import FactoryUR5eTask, GearMesh, NutThread, NutUnthread, PegInsert

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 6,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}

UR5E_INITIAL_JOINT_ANGLES = (0.0, -1.333e+00, 1.792e+00, -2.049e+00, -1.572e+00,  6.203)
UR5E_INITIAL_GRIPPER_POS = (0.015, 0.015)

@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    reset_joints = UR5E_INITIAL_JOINT_ANGLES
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    default_dof_pos_tensor = UR5E_INITIAL_JOINT_ANGLES
    kp_null = 10.0
    kd_null = 6.3246


@configclass
class FactoryUR5eEnvCfg(DirectRLEnvCfg):
    decimation: int = 8
    action_space: int = 6
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space: int = 21
    state_space: int = 72
    obs_order: list[str] = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    state_order: list[str] = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: FactoryUR5eTask = FactoryUR5eTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,  # Important for stable simulation.
            gpu_collision_stack_size=2**28
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/ur5e",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/factoryur5e/ur5e_rl_libre.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "ur5e_joint1":UR5E_INITIAL_JOINT_ANGLES[0],
                "ur5e_joint2":UR5E_INITIAL_JOINT_ANGLES[1],
                "ur5e_joint3":UR5E_INITIAL_JOINT_ANGLES[2],
                "ur5e_joint4":UR5E_INITIAL_JOINT_ANGLES[3],
                "ur5e_joint5":UR5E_INITIAL_JOINT_ANGLES[4],
                "ur5e_joint6":UR5E_INITIAL_JOINT_ANGLES[5],
                "ur5e_finger_joint1":UR5E_INITIAL_GRIPPER_POS[0], 
                "ur5e_finger_joint2":UR5E_INITIAL_GRIPPER_POS[1],
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "ur5e": ImplicitActuatorCfg(
                joint_names_expr=["ur5e_joint[1-3]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=87,
                velocity_limit=124.6,
            ),
            "ur5e_forearm": ImplicitActuatorCfg(
                joint_names_expr=["ur5e_joint[4-6]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit=12,
                velocity_limit=149.5,
            ),
            "ur5e_hand": ImplicitActuatorCfg(
                joint_names_expr=["ur5e_finger_joint.*"],
                effort_limit=40.0,
                velocity_limit=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )


@configclass
class FactoryUR5eTaskPegInsertCfg(FactoryUR5eEnvCfg):
    task_name = "peg_insert"
    task = PegInsert()
    episode_length_s = 10.0


@configclass
class FactoryUR5eTaskGearMeshCfg(FactoryUR5eEnvCfg):
    task_name = "gear_mesh"
    task = GearMesh()
    episode_length_s = 20.0


@configclass
class FactoryUR5eTaskNutThreadCfg(FactoryUR5eEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0

@configclass
class FactoryUR5eTaskNutUnthreadCfg(FactoryUR5eEnvCfg):
    task_name = "nut_unthread"
    task = NutUnthread()
    episode_length_s = 30.0
