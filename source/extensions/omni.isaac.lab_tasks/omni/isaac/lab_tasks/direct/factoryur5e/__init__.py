# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .factory_env import FactoryUR5eEnv
from .factory_env_cfg import FactoryUR5eTaskGearMeshCfg, FactoryUR5eTaskNutThreadCfg, FactoryUR5eTaskPegInsertCfg, FactoryUR5eTaskNutUnthreadCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Factory-UR5e-PegInsert-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.factoryur5e:FactoryUR5eEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryUR5eTaskPegInsertCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-UR5e-GearMesh-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.factoryur5e:FactoryUR5eEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryUR5eTaskGearMeshCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Factory-UR5e-NutThread-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.factoryur5e:FactoryUR5eEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryUR5eTaskNutThreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"
    },
)

gym.register(
    id="Isaac-Factory-UR5e-NutUnthread-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.factoryur5e:FactoryUR5eEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryUR5eTaskNutUnthreadCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml"
    },
)
