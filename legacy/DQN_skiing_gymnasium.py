import gymnasium as gym
import numpy as np
import torch
import time
import pickle
from collections import deque
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dqn.agent import Agent  # 반드시 적절한 agent.py가 있어야 함


# -----------------------
# 하이퍼파라미터 설정
# -----------------------
env_id = "ALE/Skiing-v5"
n_episodes = 1000
max_t = 10000
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.995
n_unit = 512
seed = 0
save_name = "skiing_ram"

# -----------------------
# 환경 생성 (RAM 입력)
# -----------------------
env = gym.make(env_id, obs_type="ram", render_mode=None)
obs_shape = env.observation_space.shape
action_size = env.action_space.n

print(f"Observation space: {obs_shape}, Action space: {action_size}")

# -----------------------
# Agent 생성
# -----------------------
agent = Agent(state_size=obs_shape[0], action_size=action_size, seed=seed, n_unit=n_unit)

# -----------------------
# 기록 초기화
# -----------------------
scores = []
steps = []
scores_window = deque(maxlen=10)
max_score = -np.inf

# -----------------------
# 학습 루프
# -----------------------
for i_episode in range(1, n_episodes + 1):
    state, _ = env.reset(seed=seed + i_episode)
    score = 0

    for t in range(max_t):
        action = agent.act(state, eps_start)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward
        if done:
            break

    # 탐험률 감소
    eps_start = max(eps_start * eps_decay, eps_end)

    scores.append(score)
    steps.append(t + 1)
    scores_window.append(score)

    print(f"Episode {i_episode} - Score: {score:.2f}, Steps: {t+1}, Epsilon: {eps_start:.4f}")

    # 모델 저장 (최고 점수일 때)
    if score > max_score:
        torch.save(agent.qnetwork_local.state_dict(), f"dqn/model/dqn_max_{save_name}.pth")
        max_score = score
        print(f"New max score! Model saved at Episode {i_episode}")

# -----------------------
# 학습 결과 저장
# -----------------------
data = {
    "scores": scores,
    "steps": steps
}
with open(f"dqn/score/trajectory_{save_name}.pkl", "wb") as f:
    pickle.dump(data, f)

env.close()
