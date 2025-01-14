# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5Erobots.

The following configurations are available:

* :obj:`UR5E_CFG`: UR5E robot
* :obj:`UR5E_HIGH_PD_CFG`: UR5Erobot with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
#from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os

##
# Configuration
##

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.getcwd()}/source/extensions/omni.isaac.lab_assets/omni/isaac/models/models/ur5e_rl_libre.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
                "ur5e_joint1":0.0,
                "ur5e_joint2":-1.57e+00,
                "ur5e_joint3":1.38e+00,
                "ur5e_joint4":-1.35e+00,
                "ur5e_joint5":-1.57e+00,
                "ur5e_joint6":5.60e-01,
                "ur5e_finger_joint1":0.013, 
                "ur5e_finger_joint2":0.013,
        },
    ),
    actuators={
        "ur5e": ImplicitActuatorCfg(
            joint_names_expr=["ur5e_joint[1-3]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "ur5e_forearm": ImplicitActuatorCfg(
            joint_names_expr=["ur5e_joint[4-6]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "ur5e_hand": ImplicitActuatorCfg(
            joint_names_expr=["ur5e_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of UR5E robot."""


UR5E_HIGH_PD_CFG = UR5E_CFG.copy()
UR5E_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR5E_HIGH_PD_CFG.actuators["ur5e"].stiffness = 400.0
UR5E_HIGH_PD_CFG.actuators["ur5e"].damping = 80.0
UR5E_HIGH_PD_CFG.actuators["ur5e_forearm"].stiffness = 400.0
UR5E_HIGH_PD_CFG.actuators["ur5e_forearm"].damping = 80.0
"""Configuration of UR5E robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
