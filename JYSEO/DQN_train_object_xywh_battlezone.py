import os
import time
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from ocatari.core import OCAtari
import ocatari.ram.battlezone as OCA_battlezone
from dqn.agent import Agent

# 하이퍼파라미터
n_episodes      = 1000
max_t           = 10_000
eps_start       = 1.0
eps_end         = 0.05
eps_decay       = 0.995
decay_by_action = False  # 에피소드 단위 epsilon 감소 여부
hidden_units    = 512
seed            = 0

save_dir = "/content/drive/MyDrive/Colab Notebooks/dqn/object_with_replay"
os.makedirs(save_dir, exist_ok=True)

#오브젝트(x,y,w,h) 정보 추출 함수
objects_initialized = OCA_battlezone._init_objects_ram(hud=False)
def extract_objects_xywh_from_RAM(objects_initialized, ram):
    objs = objects_initialized
    OCA_battlezone._detect_objects_ram(objs, ram, hud=False)
    vec = list(objs[0].xy)  # player x,y
    for o in objs[1:]:
        vec.extend(o.xywh)
    return np.array(vec, dtype=np.float32)

# state_dim 계산
_sample_env = OCAtari(env_name="BattleZoneNoFrameskip-v4", mode="ram", render_mode=None, buffer_window_size=1)
state_dim = len(extract_objects_xywh_from_RAM(objects_initialized, _sample_env.get_ram()))
_sample_env.close()

ram0, _ = _sample_env.reset()
state_dim = len(extract_objects_xywh_from_RAM(objects_initialized, _sample_env.get_ram()))
_sample_env.close()

# 객체 기반 환경 래퍼
class OCAtariObjectEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = OCAtari(
            env_name="BattleZoneNoFrameskip-v4",
            mode="ram",
            obs_mode="ori",         
            render_mode=None,
            buffer_window_size=1
        )
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(state_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed)
        ram = self.env.get_ram()
        obs_obj = extract_objects_xywh_from_RAM(objects_initialized, ram)
        return obs_obj, info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.get_ram()
        obs_obj = extract_objects_xywh_from_RAM(objects_initialized, ram)
        return obs_obj, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# 환경
env = OCAtariObjectEnv()
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_size  = env.observation_space.shape[0]
action_size = env.action_space.n
agent       = Agent(state_size, action_size, seed, hidden_units)

scores = []
steps  = []
eps    = eps_start

for ep in range(1, n_episodes+1):
    state, _ = env.reset()
    score = 0
    # object-based 학습: state, next_state는 obs_obj 벡터
    episode_states  = []  
    episode_actions = []  
    episode_rewards = []  
    episode_scores  = []  

    for t in range(max_t):
        action = agent.act(state, eps)
        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        agent.step(state, action, reward, next_state, done)

        raw_ram = env.env.get_ram().copy()
        episode_states.append(raw_ram)
        episode_actions.append(action)
        episode_rewards.append(reward)
        score += reward
        episode_scores.append(score)

        state = next_state
        if done:
            break

    # epsilon decay
    if not decay_by_action:
        eps = max(eps * eps_decay, eps_end)

    # replay 데이터 저장
    history = {
        "rams":    episode_states,
        "actions": episode_actions,
        "rewards": episode_rewards,
        "scores":  episode_scores
    }
    pkl_path = os.path.join(save_dir, f"history_ep{ep:04d}_score{score:.1f}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(history, f)

    # 학습 스코어 저장
    scores.append(score)
    steps.append(t+1)
    print(f"[Ep {ep:03d}] Score: {score:.1f} | Eps: {eps:.3f}")

#traj = {'scores': scores, 'steps': steps}
#with open(os.path.join(save_dir, 'trajectory.pkl'), 'wb') as f:
#    pickle.dump(traj, f)

env.close()
print("\n🎉 Object-based training complete. Replay data saved!")
