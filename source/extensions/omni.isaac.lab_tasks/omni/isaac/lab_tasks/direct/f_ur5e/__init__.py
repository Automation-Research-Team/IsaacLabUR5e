# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
UR5e-Master environment.
"""

import gymnasium as gym

from . import agents
from .ur5e_master_env import UR5eMasterEnv
from .ur5e_master_env_cfg import  MasterUR5eTaskNutThreadCfg


##
# Register Gym environments.
##

gym.register(
    id="Isaac-UR5e-Master-Direct-v0",
    entry_point=f"{__name__}.ur5e_master_env:UR5eMasterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MasterUR5eTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
