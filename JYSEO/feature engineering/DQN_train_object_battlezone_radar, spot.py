import os
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from ocatari.core import OCAtari
import ocatari.ram.battlezone as OCA_battlezone
#from dqn.agent import Agent

# 하이퍼 파라미터
n_episodes      = 500      
max_t           = 10_000
eps_start       = 1.0
eps_end         = 0.05
eps_decay       = 0.999       
hidden_units    = 512
seed            = 0

save_dir = "/content/drive/MyDrive/Colab Notebooks"
os.makedirs(save_dir, exist_ok=True)
MAX_RADAR_DIST = 200.0         # 레이더 최대 거리 (정규화용)

# feature 추출
def extract_features(objects_initialized, ram, screen_width, frame, prev_frame):
    # 객체 업데이트
    OCA_battlezone._detect_objects_ram(objects_initialized, ram, hud=True)
    px, py = objects_initialized[0].xy  # 플레이어 위치
    features = []

    # 1) 적(enemy) 정보: dist, sin(angle), cos(angle), on_screen — 최대 3명
    enemies = [o for o in objects_initialized if getattr(o, 'category', None) == 'enemy']
    for e in enemies[:3]:
        cx, cy = e.xywh[0] + e.xywh[2]/2, e.xywh[1] + e.xywh[3]/2
        dx, dy = cx - px, cy - py
        dist = np.hypot(dx, dy) / MAX_RADAR_DIST
        angle = np.arctan2(dy, dx)
        on_screen = 1.0 if (0 <= cx <= screen_width) else 0.0
        features += [dist, np.sin(angle), np.cos(angle), on_screen]
    # 패딩
    while len(features) < 3 * 4:
        features += [0.0] * 4

    # 2) 레이더(radar_blip) 정보: dist, sin(angle), cos(angle) — 최대 2개
    rad_blips = [o for o in objects_initialized if getattr(o, 'category', None) in ('radar', 'radar_blip')]
    rad_blips = sorted(rad_blips, key=lambda o: getattr(o, 'distance', MAX_RADAR_DIST))[:2]
    for r in rad_blips:
        dist = getattr(r, 'distance', 0.0) / MAX_RADAR_DIST
        angle = getattr(r, 'angle', 0.0)
        features += [dist, np.sin(angle), np.cos(angle)]
    # 패딩
    while len(features) < 3*4 + 2*3:
        features += [0.0] * 3

    # 3) spot 감지 (object 모드 활용)
    spots = [o for o in objects_initialized if getattr(o, 'category', None) == 'spot']
    spot_active = 1.0 if spots and getattr(spots[0], 'active', False) else 0.0
    features.append(spot_active)

    # 4) 프레임 차분 (센서 깜빡임 보조) — optional
    if prev_frame is not None:
        hx, hy = frame.shape[1]//2, frame.shape[0]//4
        diff = abs(int(frame[hy, hx, 0]) - int(prev_frame[hy, hx, 0]))
        flash_aux = 1.0 if diff > 30 else 0.0
    else:
        flash_aux = 0.0
    features.append(flash_aux)

    return np.array(features, dtype=np.float32), spot_active, len(rad_blips) > 0

# 환경 래퍼
class OCAtariObjectEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = OCAtari(
            env_name="BattleZoneNoFrameskip-v4",
            mode="ram",
            obs_mode="ori",
            render_mode="rgb_array",
            buffer_window_size=1
        )
        self.objects_initialized = OCA_battlezone._init_objects_ram(hud=True)
        self.screen_width = 160
        self.prev_frame = None

        # state_dim 계산
        _, _         = self.env.reset()
        sample_ram   = self.env.get_ram()
        sample_frame = self.env.render()
        feat_vec, _, _ = extract_features(
            self.objects_initialized,
            sample_ram,
            self.screen_width,
            sample_frame,
            None
        )
        self.state_dim = len(feat_vec)

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        _, info  = self.env.reset(seed=seed)
        ram       = self.env.get_ram()
        frame     = self.env.render()
        obs, _, _ = extract_features(
                        self.objects_initialized,
                        ram,
                        self.screen_width,
                        frame,
                        None
                    )
        self.prev_frame = frame.copy()
        return obs, info

    def step(self, action):
        _, reward, term, trunc, info = self.env.step(action)
        ram       = self.env.get_ram()
        frame     = self.env.render()
        obs, spot_active, radar_flag = extract_features(
                        self.objects_initialized,
                        ram,
                        self.screen_width,
                        frame,
                        self.prev_frame
                    )
        self.prev_frame = frame.copy()
        done = term or trunc

        # 보상 보너스
        reward_shaped = reward
        # 1) spot 켜질 때 보너스
        reward_shaped += 0.1 * spot_active
        # 2) 레이더 블립 감지 후 적 반응할 때 보너스
        reward_shaped += 0.05 * float(radar_flag and spot_active)

        return obs, reward_shaped, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()

# 학습 루프
env = OCAtariObjectEnv()
np.random.seed(seed)
torch.manual_seed(seed)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_size  = env.state_dim
action_size = env.action_space.n
agent       = Agent(state_size, action_size, seed, hidden_units)

scores, steps, eps = [], [], eps_start
for ep in range(1, n_episodes+1):
    state, _ = env.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state, eps)
        next_state, reward, done, info = env.step(action)
        agent.step(state, action, reward, next_state, done)

        score += reward
        state  = next_state
        if done:
            break

    eps = max(eps * eps_decay, eps_end)

    with open(os.path.join(save_dir, f"history_ep{ep:04d}_score{score:.1f}.pkl"), 'wb') as f:
        pickle.dump({
            "rams": env.env.get_ram(),
            "scores": [score]
        }, f)

    scores.append(score)
    steps.append(t+1)
    print(f"[Ep {ep:03d}] Score: {score:.1f} | Eps: {eps:.3f}")

# 전체 결과 저장
with open(os.path.join(save_dir, 'trajectory.pkl'), 'wb') as f:
    pickle.dump({'scores': scores, 'steps': steps}, f)

env.close()
print("🎉 Training complete!")
