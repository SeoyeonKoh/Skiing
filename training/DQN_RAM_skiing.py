from ocatari.core import OCAtari
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import time
import pickle
import argparse
from dqn.agent import Agent

# 학습 하이퍼파라미터
n_episodes = 1700
max_t = 10000
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.995
decay_by_action = False

n_unit = 512
seed = 0
n_iter = 1
save_name = 'skiing_ocatari'

# setup 저장용 구조체
dqn_setup = argparse.Namespace()
dqn_setup.eps_start = eps_start
dqn_setup.eps_decay = eps_decay
dqn_setup.eps_end = eps_end
dqn_setup.max_t = max_t
dqn_setup.n_unit = n_unit
dqn_setup.version = 1
dqn_setup.max_speed = None
dqn_setup.decay_by_action = decay_by_action

# 점수 기록
scores = []
steps = []
scores_window = deque(maxlen=10)
max_score = 0
max_by_iter = 0

# 환경 래퍼 클래스
class OCAtariWrapper(gym.Env):
    def __init__(self, env_name, mode, obs_mode, render_mode, buffer_window_size):
        super(OCAtariWrapper, self).__init__()
        self.env = OCAtari(env_name=env_name, mode=mode, obs_mode=obs_mode,
                           render_mode=render_mode, buffer_window_size=buffer_window_size)
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(128,), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()
        return self.env.get_ram()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.env.get_ram(), reward, terminated, truncated, info

    def get_ram(self):
        return self.env.get_ram()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# 여기에서 Skiing 환경으로 설정
env = OCAtariWrapper(env_name="Skiing", mode="ram",
                     obs_mode="ori", render_mode="rgb_array", buffer_window_size=1)

# 학습 반복
for i_iter in range(n_iter):
    print("iteration", i_iter, "start")
    sim_start_time = time.time()

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = Agent(state_size=obs_shape[0], action_size=action_size, seed=seed, n_unit=n_unit)

    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0

        episode_states = []
        episode_actions = []
        episode_rewards = []

        for t in range(max_t):
            action = agent.act(state, eps)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)

            episode_states.append(state.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state
            score += reward

            if done:
                break

        print(f"agent#{i_episode} scored {score}")

        # 저장
        record_history = {
            "states": episode_states,
            "actions": episode_actions,
            "rewards": episode_rewards
        }

        with open(f'dqn/history{i_episode}_score{score}.pkl', 'wb') as f:
            pickle.dump(record_history, f)

        scores_window.append(score)
        scores.append(score)
        steps.append(t + 1)
        eps = max(eps * eps_decay, eps_end)

    print("Iteration complete. Time elapsed:", time.time() - sim_start_time)

# 최종 trajectory 저장
data = {
    'steps': steps,
    'scores': scores
}

with open('trajectory.pkl', 'wb') as f:
    pickle.dump(data, f)
