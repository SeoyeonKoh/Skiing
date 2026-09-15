# DQN_skiing.py
# - Skiing (RAM) + DQN 학습 스크립트
# - 폴더: ~/Desktop/Skiing
#   └─ dqn/agent.py, dqn/model.py, dqn/__init__.py, dqn/model/, dqn/score/



from gymnasium.wrappers import RecordVideo
import os, time
import numpy as np

import os
import time
import pickle
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from dqn.agent import Agent  # Agent(state_size, action_size, seed, n_unit)

# -----------------------
# 설정
# -----------------------
ENV_ID = "ALE/Skiing-v5"
OBS_TYPE = "ram"          # (128,) RAM 관측
N_EPISODES = 1000         # 먼저는 50으로 테스트 권장
MAX_T = 10000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

N_UNIT = 512
SEED = 0
SAVE_NAME = "skiing_ram"

# 저장 폴더 생성
os.makedirs("dqn/model", exist_ok=True)
os.makedirs("dqn/score", exist_ok=True)

# -----------------------
# 환경/에이전트 준비
# -----------------------
env = gym.make(ENV_ID, obs_type=OBS_TYPE, render_mode=None)
obs_space = env.observation_space
act_space = env.action_space

# 안전 확인
assert len(obs_space.shape) == 1 and obs_space.shape[0] == 128, f"RAM shape mismatch: {obs_space}"
assert hasattr(act_space, "n") and act_space.n == 3, f"Action space mismatch: {act_space}"

print(f"[env] obs={obs_space.shape}, actions={act_space.n}  (0:NOOP, 1:RIGHT, 2:LEFT)")

agent = Agent(state_size=128, action_size=3, seed=SEED, n_unit=N_UNIT)

# -----------------------
# 학습 루프
# -----------------------
scores = []
steps = []
scores_window = deque(maxlen=10)
best_score = -np.inf
eps = EPS_START
t0 = time.time()

for i_ep in range(1, N_EPISODES + 1):
    state, _ = env.reset(seed=SEED + i_ep)
    # 선택: RAM을 0~1로 스케일 (agent 내부에서 처리 안 하면 여기서)
    state = state.astype(np.float32) / 255.0

    ep_score = 0.0

    for t in range(MAX_T):
        action = agent.act(state, eps)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = next_state.astype(np.float32) / 255.0

        agent.step(state, action, reward, next_state, done)

        state = next_state
        ep_score += reward
        if done:
            break

    # epsilon decay
    eps = max(eps * EPS_DECAY, EPS_END)

    scores.append(ep_score)
    steps.append(t + 1)
    scores_window.append(ep_score)

    print(
        f"Ep {i_ep:4d} | score {ep_score:8.2f} | steps {t+1:5d} | "
        f"eps {eps: .4f} | avg10 {np.mean(scores_window):8.2f}"
    )

    # 최고 점수 갱신 시 모델 저장 (Skiing은 덜 음수가 '더 좋음' → 큰 값이 더 좋음)
    if ep_score > best_score:
        best_score = ep_score
        torch.save(agent.qnetwork_local.state_dict(), f"dqn/model/dqn_max_{SAVE_NAME}.pth")
        print(f"  -> new best {best_score:.2f} (saved)")

# -----------------------
# 결과 저장
# -----------------------
with open(f"dqn/score/trajectory_{SAVE_NAME}.pkl", "wb") as f:
    pickle.dump({"scores": scores, "steps": steps}, f)

env.close()
print(f"[done] episodes={N_EPISODES}, elapsed={time.time()-t0:.1f}s, best={best_score:.2f}")
