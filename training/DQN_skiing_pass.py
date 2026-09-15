# ===== Skiing (OCAtari RAM) + Reward Shaping (collision / gate miss / stuck) =====

import os
import ale_py
from ocatari.core import OCAtari
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import numpy as np
from collections import deque
import time
import pickle
import argparse
from typing import List, Dict, Optional
from dqn.agent import Agent

n_episodes = 3600
max_t = 10000
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.995
decay_by_action = False

n_unit = 512
seed = 0
n_iter = 1
save_name = 'skiing_ocatari_ram'

dqn_setup = argparse.Namespace()
dqn_setup.eps_start = eps_start
dqn_setup.eps_decay = eps_decay
dqn_setup.eps_end = eps_end
dqn_setup.max_t = max_t
dqn_setup.n_unit = n_unit
dqn_setup.version = 1
dqn_setup.max_speed = None
dqn_setup.decay_by_action = decay_by_action

scores = []
steps = []
scores_window = deque(maxlen=10)
max_score = 0
max_by_iter = 0


class OCAtariRAMWrapper(gym.Env):
    def __init__(self, env_name, mode="ram", render_mode="rgb_array", stack_frames=1):
        super().__init__()
        self.env = OCAtari(env_name=env_name, mode=mode, obs_mode="obj",
                           render_mode=render_mode, buffer_window_size=1)
        self.action_space = self.env.action_space
        self.stack_frames = stack_frames
        self._ram_stack = deque(maxlen=stack_frames)
        self.observation_space = spaces.Box(low=0, high=255, shape=(128 * stack_frames,), dtype=np.uint8)

    def _get_ram_vec(self):
        return np.asarray(self.env.get_ram(), dtype=np.uint8)

    def _stacked_obs(self):
        if self.stack_frames == 1:
            return self._ram_stack[0]
        return np.concatenate(list(self._ram_stack), axis=0)

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self._ram_stack.clear()
        ram = self._get_ram_vec()
        for _ in range(self._ram_stack.maxlen):
            self._ram_stack.append(ram)
        return self._stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self._get_ram_vec()
        self._ram_stack.append(ram)
        return self._stacked_obs(), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    @property
    def objects(self):
        return getattr(self.env, "objects", None)

    def get_objects(self):
        fn = getattr(self.env, "get_objects", None)
        return fn() if callable(fn) else None


def _oget(o, key, default=0):
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)

def _bbox_overlap(a, b):
    ax, ay = _oget(a, "x", 0), _oget(a, "y", 0)
    aw, ah = _oget(a, "w", 2) or 2, _oget(a, "h", 2) or 2
    bx, by = _oget(b, "x", 0), _oget(b, "y", 0)
    bw, bh = _oget(b, "w", 2) or 2, _oget(b, "h", 2) or 2
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


class OCObjectsHelperMixin:
    def _unwrap_for_objects(self):
        ref = self.env
        seen = 0
        while ref is not None and seen < 6:
            if hasattr(ref, "objects") or hasattr(ref, "get_objects"):
                return ref
            ref = getattr(ref, "env", None)
            seen += 1
        return self.env

    def _get_objects(self) -> Optional[List[Dict]]:
        base = self._unwrap_for_objects()
        objs = None
        if hasattr(base, "objects") and base.objects is not None:
            if isinstance(base.objects, list):
                objs = base.objects
            elif isinstance(base.objects, dict):
                flat = []
                for _k, _v in base.objects.items():
                    if isinstance(_v, list):
                        flat.extend(_v)
                objs = flat if flat else None
        if objs is None and hasattr(base, "get_objects"):
            try:
                ret = base.get_objects()
                if isinstance(ret, list) and ret:
                    objs = ret
            except Exception:
                pass
        return objs

    def _find_player(self, objs) -> Optional[dict]:
        if not objs:
            return None
        cands = [o for o in objs if str(_oget(o, "name", "")).lower() in
                 ["player", "skier", "agent", "player1", "player_1"]]
        if cands:
            return max(cands, key=lambda o: _oget(o, "y", -1))
        return max(objs, key=lambda o: _oget(o, "y", -1))

    def _find_trees(self, objs) -> List[dict]:
        if not objs:
            return []
        names = {"tree", "pine", "obstacle", "tree1", "tree2", "stump", "rock"}
        return [o for o in objs if str(_oget(o, "name", "")).lower() in names]

    def _find_flags(self, objs) -> List[dict]:
        if not objs:
            return []
        names = {"flag", "gate", "left_flag", "right_flag", "pole", "flag_left", "flag_right"}
        return [o for o in objs if str(_oget(o, "name", "")).lower() in names]


class RewardShapingWrapper(gym.Wrapper, OCObjectsHelperMixin):
    def __init__(self, env,
                 collision_penalty=-0.5,
                 miss_penalty=-0.1,
                 stuck_penalty=-0.01,
                 stuck_window=15,
                 min_dy=1,
                 gate_margin=2,
                 gate_band=3):
        super().__init__(env)
        self.collision_penalty = collision_penalty
        self.miss_penalty = miss_penalty
        self.stuck_penalty = stuck_penalty
        self.stuck_window = stuck_window
        self.min_dy = min_dy
        self.gate_margin = gate_margin
        self.gate_band = gate_band
        self._y_hist = deque(maxlen=stuck_window)
        self._last_player = None
        self._last_objs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._y_hist.clear()
        self._last_player = None
        self._last_objs = None
        objs = self._get_objects()
        ply = self._find_player(objs) if objs else None
        if ply:
            self._y_hist.append(_oget(ply, "y", 0))
        self._last_player = ply
        self._last_objs = objs
        return obs, info

    def _penalize_collision(self, player, trees) -> float:
        if not player or not trees:
            return 0.0
        for tr in trees:
            if _bbox_overlap(player, tr):
                return self.collision_penalty
        if self._last_player is not None:
            if _oget(player, "y", 0) < _oget(self._last_player, "y", 0) - 5:
                return self.collision_penalty
        return 0.0

    def _penalize_miss_gate(self, player, flags) -> float:
        if not player or not flags:
            return 0.0
        py = _oget(player, "y", 0)
        near = [f for f in flags if abs(_oget(f, "y", 0) - py) <= self.gate_band]
        if len(near) < 2:
            return 0.0
        near = sorted(near, key=lambda f: _oget(f, "x", 0))
        left_flag, right_flag = near[0], near[-1]
        gate_left = _oget(left_flag, "x", 0) - self.gate_margin
        gate_right = _oget(right_flag, "x", 0) + (_oget(right_flag, "w", 0) or 0) + self.gate_margin
        px = _oget(player, "x", 0) + (_oget(player, "w", 2) or 2) / 2.0
        if not (gate_left <= px <= gate_right):
            return self.miss_penalty
        return 0.0

    def _penalize_stuck(self, player) -> float:
        if not player:
            return 0.0
        self._y_hist.append(_oget(player, "y", 0))
        if len(self._y_hist) < self._y_hist.maxlen:
            return 0.0
        dy = self._y_hist[-1] - self._y_hist[0]
        if dy < self.min_dy:
            return self.stuck_penalty
        return 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0
        objs = self._get_objects()
        player = self._find_player(objs) if objs else None
        trees = self._find_trees(objs) if objs else []
        flags = self._find_flags(objs) if objs else []
        shaped += self._penalize_collision(player, trees)
        shaped += self._penalize_miss_gate(player, flags)
        shaped += self._penalize_stuck(player)
        total_reward = float(np.clip(reward + shaped, -1.0, 1.0))
        if info is None:
            info = {}
        info["shaping"] = shaped
        self._last_player = player
        self._last_objs = objs
        return obs, total_reward, terminated, truncated, info

base_env = OCAtariRAMWrapper(env_name="ALE/Skiing-v5", mode="ram", render_mode="rgb_array", stack_frames=1)
shaped_env = RewardShapingWrapper(
    base_env,
    collision_penalty=-0.5,
    miss_penalty=-0.1,
    stuck_penalty=-0.01,
    stuck_window=15,
    min_dy=1,
    gate_margin=2,
    gate_band=3
)
env = TimeLimit(shaped_env, max_episode_steps=max_t)

record_history = {"states": [], "actions": [], "rewards": []}
os.makedirs("dqn", exist_ok=True)

for i_iter in range(n_iter):
    print("iteration", i_iter, "start")
    sim_start_time = time.time()

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    agent = Agent(state_size=obs_shape[0], action_size=action_size, seed=seed, n_unit=n_unit)

    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        if i_episode == 1:
            state, _ = env.reset(seed=seed)
        else:
            state, _ = env.reset()
        score = 0
        episode_states, episode_actions, episode_rewards = [], [], []

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            is_last_step = (t == max_t - 1)
            done = terminated or truncated or is_last_step

            agent.step(state, action, reward, next_state, done)

            episode_states.append(state.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state
            score += reward
            if done:
                break

        print(f"episode {i_episode} score {score}")
        record_history["states"] = episode_states
        record_history["actions"] = episode_actions
        record_history["rewards"] = episode_rewards

        with open(f'dqn/history{i_episode}_score{score}.pkl', 'wb') as f:
            pickle.dump(record_history, f)

        scores_window.append(score)
        scores.append(score)
        steps.append(t + 1)
        eps = max(eps * eps_decay, eps_end)

data = {'steps': steps, 'scores': scores}
with open('trajectory.pkl', 'wb') as f:
    pickle.dump(data, f)
