import os
import time
import pickle
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

import torch
from ocatari.core import OCAtari
from dqn.agent import Agent


# ─── 학습 하이퍼파라미터 ───────────────────────────────────────────────
n_episodes       = 2000
max_t            = 10000
eps_start        = 1.0
eps_end          = 0.05
eps_decay        = 0.995
decay_by_action  = False

n_unit           = 512
seed             = 0
n_iter           = 1
save_name        = 'battlezone_ram'

# ─── setup 저장용 구조체 ───────────────────────────────────────────────
dqn_setup = argparse.Namespace()
dqn_setup.eps_start       = eps_start
dqn_setup.eps_decay       = eps_decay
dqn_setup.eps_end         = eps_end
dqn_setup.max_t           = max_t
dqn_setup.n_unit          = n_unit
dqn_setup.version         = 1
dqn_setup.max_speed       = None
dqn_setup.decay_by_action = decay_by_action

# ─── 결과 기록용 구조체 ───────────────────────────────────────────────
scores        = []
steps         = []
scores_window = deque(maxlen=10)

# ─── replay 기록용 기본 dict ──────────────────────────────────────────
record_history = {
    "states": [],
    "actions": [],
    "rewards": []
}

# ─── 환경 래퍼 (RAM만 뽑아서 반환) ────────────────────────────────────
class OCAtariRamEnv(gym.Env):
    def __init__(self, env_name: str, buffer_window_size: int = 1):
        super().__init__()
        self.env = OCAtari(
            env_name=env_name,
            mode="ram",
            obs_mode="ori",
            render_mode=None,
            buffer_window_size=buffer_window_size
        )
        self.action_space      = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=(128,), dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        _, info = self.env.reset(seed=seed)
        ram = np.array(self.env.get_ram(), dtype=np.uint8)
        return ram, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        ram = np.array(self.env.get_ram(), dtype=np.uint8)
        return ram, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

# ─── 환경 생성 ─────────────────────────────────────────────────────────
env = OCAtariRamEnv(env_name="BattleZoneNoFrameskip-v4", buffer_window_size=1)

# ─── 반복 학습 + replay 기록 루프 ─────────────────────────────────────
for i_iter in range(n_iter):
    print(f"\n=== iteration {i_iter} start ===")
    sim_start_time = time.time()

    # Agent 초기화
    state_size  = env.observation_space.shape[0]  # 128
    action_size = env.action_space.n
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        seed=seed,
        n_unit=n_unit
    )

    eps = eps_start
    for ep in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed)
        score = 0

        # 이번 에피소드용 기록 리스트
        episode_states  = []
        episode_actions = []
        episode_rewards = []

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)

            # replay 기록
            episode_states.append(state.copy())
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state
            score += reward
            if done:
                break

        # 에피소드 끝: 출력 및 기록
        print(f"Episode {ep:4d} | Score {score:6.1f} | Eps {eps:.3f}")

        scores_window.append(score)
        scores.append(score)
        steps.append(t + 1)

        # eps 감소 (에피소드 단위)
        if not decay_by_action:
            eps = max(eps * eps_decay, eps_end)

        # ▶▶ 저장 경로 설정 (Google Drive)
        drive_save_dir = "/content/drive/MyDrive/Colab Notebooks/dqn/RAM2"
        os.makedirs(drive_save_dir, exist_ok=True)

        save_path = os.path.join(drive_save_dir, f"history_ep{ep:04d}_score{score:.1f}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({
                "states": episode_states,
                "actions": episode_actions,
                "rewards": episode_rewards
            }, f)

        # 중간 모델 체크포인트 (Google Drive에 저장)
        if ep % 100 == 0:
            model_path = os.path.join(drive_save_dir, f"{save_name}_ckpt_ep{ep}.pth")
            torch.save(agent.q_local.state_dict(), model_path)

# ─── 전체 결과 저장 (루프 밖) ───────────────────────────────────────────
with open(os.path.join(drive_save_dir, 'trajectory.pkl'), 'wb') as f:
    pickle.dump({"scores": scores, "steps": steps}, f)

print("\nTraining complete!")
