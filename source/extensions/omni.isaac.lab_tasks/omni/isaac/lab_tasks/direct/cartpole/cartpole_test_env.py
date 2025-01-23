from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform


@configclass
class CartpoleTestEnvCfg(DirectRLEnvCfg):
    #env 
    decimation = 2
    episode_length_s = 1.0
    action_scale = 100.0
    action_space = 1
    observation_space= 2
    state_space = 0

    #simulation
    sim: SimulationCfg = SimulationCfg(dt= 1 / 120, render_interval=decimation)

    #robot
    robot_cfg = ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    #scene
    scene : InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=1.0, replicate_physics=True)

    #REset
    max_cart_pos = 10.0 #meters
    initial_pole_angle_range = [-0.55, 0.55] #rad

    #reward scales
    rew_scale_alive = 0.2
    rew_scale_terminated = -5.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pol_vel = -0.005

class CartpoleTestEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg : CartpoleTestEnvCfg

    def __init__(self, cfg: CartpoleTestEnvCfg, render_mode : str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self.pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        #add a ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color= (22,211,32)))
        #clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        #add articulation to scene
        self.scene.articulations["cartpoleSEB"] = self.cartpole
        #add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0,1.0,1.0))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.cartpole.set_joint_position_target(self.actions, joint_ids=self.cart_dof_idx)

    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    
    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self.pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self.pole_dof_idx[0]].unsqueeze(dim=1),
            ), 
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.reset_terminated
        )
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self.cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self.pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out
    
    def _reset_idx(self, env_ids:Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self.pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self.pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)



    

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    reset_terminated: torch.Tensor,
):
    total_reward = rew_scale_alive * (1.0 - reset_terminated.float())

    return total_reward
