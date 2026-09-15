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
        self.env = OCAtari(env_name=env_name, mode=mode, obs_mode=None, render_mode=render_mode, buffer_window_size=1)
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

base_env = OCAtariRAMWrapper(env_name="ALE/Skiing-v5", mode="ram", render_mode="rgb_array", stack_frames=1)
env = TimeLimit(base_env, max_episode_steps=max_t)

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

        episode_states = []
        episode_actions = []
        episode_rewards = []

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            clipped_reward = float(np.clip(reward, -1.0, 1.0))
            is_last_step = (t == max_t - 1)
            done = terminated or truncated or is_last_step

            agent.step(state, action, clipped_reward, next_state, done)

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
