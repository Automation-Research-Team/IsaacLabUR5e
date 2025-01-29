# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.utils import configclass

from .factory_tasks_cfg import ASSET_DIR, FactoryTask, NutThread

#OBS_DIM_CFG = {
#    "fingertip_pos": 3,
#    "fingertip_pos_rel_fixed": 3,
#    "fingertip_quat": 4,
#    "ee_linvel": 3,
#    "ee_angvel": 3,
#}

#STATE_DIM_CFG = {
#    "fingertip_pos": 3,
#    "fingertip_pos_rel_fixed": 3,
#    "fingertip_quat": 4,
#    "ee_linvel": 3,
#    "ee_angvel": 3,
#    "joint_pos": 7,
#    "held_pos": 3,
#    "held_pos_rel_fixed": 3,
#    "held_quat": 4,
#    "fixed_pos": 3,
#    "fixed_quat": 4,
#    #"task_prop_gains": 6,
#    "ema_factor": 1,
#    "pos_threshold": 3,
#    "rot_threshold": 3,
#}


OBS_DIM_CFG = {
    "fingertip_pos_rel_fixed": 3,
    "Pos_error": 3,
    "Axis_error": 3,
    "error_ee_linvel": 3,
    "error_ee_angvel": 3,
    "tool_Force": 3,
    "tool_Torque":3,
}

STATE_DIM_CFG = {
    "Pos_error": 3,
    "Axis_error": 3,
    "error_ee_linvel": 3,
    "error_ee_angvel": 3,
    "tool_Force": 3,
    "tool_Torque":3,
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 9,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]
    
    gripper_pos_action_bounds = 0.008

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]
    
    reset_joints = [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]
    #reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    #reset_rot_deriv_scale = 10.0
    #default_task_prop_gains = [100, 100, 100, 30, 30, 30]

    # Null space parameters.
    #default_dof_pos_tensor = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    #kp_null = 10.0
    #kd_null = 6.3246


@configclass
class FactoryEnvCfg(DirectRLEnvCfg):
    decimation: int = 8
    action_space: int = 7
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space: int = 1
    state_space: int = 7
    #obs_order: list[str] = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel", "ee_angvel"]
    #state_order: list[str] = [
    #    "fingertip_pos",
    #    "fingertip_quat",
    #    "ee_linvel",
    #    "ee_angvel",
    #    "joint_pos",
    #    "held_pos",
    #    "held_pos_rel_fixed",
    #    "held_quat",
    #    "fixed_pos",
    #    "fixed_quat",
    #]
    obs_order: list[str] = ["fingertip_pos_rel_fixed", "Pos_error", "Axis_error", "error_ee_linvel", "error_ee_angvel", "tool_Force", "tool_Torque"] 
    state_order: list[str] = [
        "Pos_error", 
        "Axis_error", 
        "error_ee_linvel", 
        "error_ee_angvel", 
        "tool_Force", 
        "tool_Torque",
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

    task_name: str = "nut_thread"  # peg_insert, gear_mesh, nut_thread
    task: FactoryTask = FactoryTask()
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
            gpu_max_rigid_contact_count=2**25,
            gpu_max_rigid_patch_count=2**25,
            gpu_found_lost_pairs_capacity = 2**23,
            gpu_collision_stack_size = 5*2**26,
            gpu_heap_capacity = 2**28,
            gpu_temp_buffer_capacity = 2**28,
            gpu_max_particle_contacts= 2**21,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/franka_mimic.usd",
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
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness= 300, #400, #7500.0,
                damping= 35 ,# 80, #173.0,
                friction=0.1,
                armature=0.0,
                effort_limit=87.0,
                velocity_limit=2.175,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness= 300 , #400, #7500.0,
                damping= 35 , # 80, #173.0,
                friction=0.1,
                armature=0.0,
                effort_limit=12.0,
                velocity_limit=2.61,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit=200,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
                friction=0.1,
                armature=0.0,
            ),
        },
        soft_joint_pos_limit_factor=1.0
    )
    robot_forearm_to_base_frame_transformer: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/panda_fingertip_centered",            #panda_link7",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/Robot/panda_link0")],
        debug_vis=False,
    )

@configclass
class FactoryTaskNutThreadCfg(FactoryEnvCfg):
    task_name = "nut_thread"
    task = NutThread()
    episode_length_s = 30.0
