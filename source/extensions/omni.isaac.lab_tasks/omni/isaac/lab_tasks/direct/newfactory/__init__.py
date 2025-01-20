# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .factory_env import FactoryEnv
from .factory_env_cfg import FactoryTaskNutThreadCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-newFactory-NutThread-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.newfactory:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
