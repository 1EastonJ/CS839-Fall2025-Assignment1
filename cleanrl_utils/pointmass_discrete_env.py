import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register as gym_register
from cleanrl_utils.pointmass_continuous_env import PointMassContinuousEnv


class PointMassDiscreteEnv(PointMassContinuousEnv):
    """A continuous point-mass agent in a 2D plane with obstacles.

    - Observation: continuous vector [agent_x, agent_y, goal_x, goal_y]
    - Action space: Discrete(4) with 4 movement directions
    - Reward: distance-based shaping, goal bonus, small penalty for collisions
    - Success metric (logged only): time spent at goal − number of hits
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        super().__init__(seed=seed, render_mode=render_mode)
        self.render_mode = render_mode

        # Discrete action space (4 directions)
        self.action_space = spaces.Discrete(4)
        self._actions = {
            0: np.array([+2, 0], dtype=np.float32),   # move right
            1: np.array([-2, 0], dtype=np.float32),   # move left
            2: np.array([0, +2], dtype=np.float32),   # move up
            3: np.array([0, -2], dtype=np.float32),   # move down
        }

        # self.observation_space = spaces.Box(
        #     low=0.0, high=1.0, shape=(4,), dtype=np.float32
        # )

        num_obstacles = len(self._get_obstacles())
        obs_dim = 4 + num_obstacles * 4  # agent+goal + obstacles
        self.observation_space = spaces.Box(
            low=0.0,
            high=max(self.world_width, self.world_height),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Disable obstacles for simplicity
        # self._get_obstacles = lambda: []

        # Initialize environment state
        self._reset_positions()

        # Initialize counters for performance metric
        self._steps = 0
        self._at_goal_steps = 0
        self._hit_obstacles = 0


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment state and all episode-specific counters."""
        super().reset(seed=seed, options=options)

        self._steps = 0
        self._at_goal_steps = 0
        self._hit_obstacles = 0

        self._reset_positions()

        observation = self._get_obs()
        info = {}
        return observation, info


    def step(self, action):
        assert self.action_space.contains(action)

        control = self._actions[int(action)]
        old_distance = np.linalg.norm(self._agent_pos - self._goal_pos)
        new_pos = self._agent_pos + control * 0.1

        terminated = False
        truncated = False
        reward = 0.0

        # --- Penalty for being too close to obstacles ---
        safe_distance = 0.45  # threshold for how close counts as "too close"
        min_dist_to_obstacle = float('inf')

        for (x1, y1, x2, y2) in self._get_obstacles():
            # Find the closest point on the obstacle rectangle to the agent
            closest_x = np.clip(new_pos[0], x1, x2)
            closest_y = np.clip(new_pos[1], y1, y2)
            dist = np.linalg.norm(new_pos - np.array([closest_x, closest_y]))
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)

        # Apply a mild penalty that grows the closer the agent gets
        if min_dist_to_obstacle < safe_distance:
            # Linearly scaled penalty (stronger near obstacle)
            reward -= (safe_distance - min_dist_to_obstacle) * 50.0



        # --- 检查是否越界或碰撞障碍 ---
        if self._check_out_of_bounds(new_pos, self.agent_radius) or self._check_collision(new_pos, self.agent_radius):
            reward = -5.0
            self._hit_obstacles += 1
            # 保持旧距离用于后续计算稳定性
            new_distance = old_distance
        else:
            # --- 更新位置 ---
            self._agent_pos = new_pos
            new_distance = np.linalg.norm(self._agent_pos - self._goal_pos)

            # --- 基于距离的奖励 ---
            reward += (old_distance - new_distance) * 50

            # --- 检查是否到达目标 ---
            if new_distance < self.goal_radius * 1.5:
                reward += 50.0
                self._at_goal_steps += 1
                terminated = True

        # --- 每步时间惩罚 ---
        reward -= 0.5

        # --- 在目标附近加入轻微阻尼，避免震荡 ---
        if new_distance < self.goal_radius * 1.7:
            reward -= 0.1 * new_distance  # 轻柔衰减而不是强惩罚

        # --- 步数限制 ---
        self._steps += 1
        if self._steps >= self.max_steps:
            truncated = True

        # --- 更新观测 ---
        observation = self._get_obs()

        # --- 计算成功指标（仅日志用） ---
        if terminated:  # reached the goal
            actual_performance = (self.max_steps - self._steps) - self._hit_obstacles
        else:
            actual_performance = -self._hit_obstacles
        info = {
            "agent_pos": self._agent_pos.copy(),
            "goal_pos": self._goal_pos.copy(),
            "distance_to_goal": float(new_distance),
            "actual_performance": actual_performance,
        }

        return observation, reward, terminated, truncated, info



    # def _get_obs(self) -> np.ndarray:
    #     """Return normalized observation."""
    #     agent_x = self._agent_pos[0] / self.world_width
    #     agent_y = self._agent_pos[1] / self.world_height
    #     goal_x = self._goal_pos[0] / self.world_width
    #     goal_y = self._goal_pos[1] / self.world_height
    #     return np.array([agent_x, agent_y, goal_x, goal_y], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Return observation including absolute obstacle coordinates."""
        agent_x = self._agent_pos[0]
        agent_y = self._agent_pos[1]
        goal_x = self._goal_pos[0]
        goal_y = self._goal_pos[1]

        # Flatten all obstacle rectangles into one array
        obstacles_flat = np.array(
            [coord for rect in self._get_obstacles() for coord in rect],
            dtype=np.float32,
        )

        obs = np.concatenate(([agent_x, agent_y, goal_x, goal_y], obstacles_flat))
        return obs



    def close(self):
        pass


def make_env(
    world_width: float = 10.0,
    world_height: float = 10.0,
    agent_radius: float = 0.2,
    goal_radius: float = 0.3,
    max_velocity: float = 2.0,
    max_episode_steps: int = 500,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> PointMassDiscreteEnv:
    return PointMassDiscreteEnv(seed=seed, render_mode=render_mode)


# --- Gymnasium registration helpers ---
DEFAULT_ENV_ID = "PointMassDiscrete-v0"


def register_pointmass_discrete_env(env_id: str = DEFAULT_ENV_ID, **kwargs) -> None:
    """Register the PointMassDiscreteEnv with Gymnasium."""
    gym_register(
        id=env_id,
        entry_point="cleanrl_utils.pointmass_discrete_env:PointMassDiscreteEnv",
        kwargs=kwargs,
        max_episode_steps=None,
    )
