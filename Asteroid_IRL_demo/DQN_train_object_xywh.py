from ocatari.core import OCAtari
import ocatari.ram.asteroids as OCA_asteroids
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import pickle
from dqn.agent import Agent

# 학습 하이퍼파라미터
n_episodes = 30
max_t = 10000
eps_start = 1.0
eps_end = 0.05
eps_decay = 0.995
decay_by_action = False

n_unit = 512
seed = 0
n_iter = 1

# trajectory 기록
scores = []
steps = []

# extract object info from RAM, 0~2: player x, y, orientation, 3~122: asteroids xywh, 123~130: missiles xywh
def extract_objects_xywh_from_RAM(objects_initialized,RAM):
    objects = objects_initialized
    OCA_asteroids._detect_objects_ram(objects, RAM, hud=False)
    obs_objects = objects[0]._nsrepr #player x, y, orientation
    for i in range(32):
        obs_objects.extend((objects[i+1].xy + objects[i+1].wh)) # 1~30: asteroids, 31~32: missiles
    return obs_objects

# extract object info from RAM, 0~2: player x, y, orientation, asteroids xy, missiles xy
def extract_objects_xy_from_RAM(objects_initialized,RAM):
    objects = objects_initialized
    OCA_asteroids._detect_objects_ram(objects, RAM, hud=False)
    obs_objects = objects[0]._nsrepr #player x, y, orientation
    for i in range(32):
        obs_objects.extend((objects[i+1].xy)) # 1~30: asteroids, 31~32: missiles
    return obs_objects

# environment
objects_initialized = OCA_asteroids._init_objects_ram(hud=False)
obs_shape=(67,)
class OCAtariWrapper(gym.Env):
    def __init__(self, env_name, mode, obs_mode, render_mode, buffer_window_size):
        super(OCAtariWrapper, self).__init__()
        self.env = OCAtari(env_name=env_name, mode=mode, obs_mode=obs_mode, render_mode=render_mode, buffer_window_size=buffer_window_size)
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
        a = self.env.reset()
        obs = (extract_objects_xy_from_RAM(objects_initialized,a[0]),a[1])
        return obs

    def step(self, action):
        obs_ram, reward, terminated, truncated, info = self.env.step(action)
        obs_objects = extract_objects_xy_from_RAM(objects_initialized,self.env.get_ram())
        return obs_ram, obs_objects, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

# Create the environment
env = OCAtariWrapper(env_name="Asteroids-ramNoFrameskip-v4", mode="ram", obs_mode="ori", render_mode="rgb_array", buffer_window_size=1)

#records for generating replay
record_history = {
    "states": [],
    "actions": [],
    "rewards": []
}

for i_iter in range(n_iter):
    print("iteration", i_iter, "start")
    sim_start_time = time.time()

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    agent = Agent(state_size=obs_shape[0], action_size=action_size, seed=seed, n_unit=n_unit)

    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        state = np.array(obs, dtype=np.uint32)
        score = 0

        #records for generating replay
        episode_states = []
        episode_rewards = []
        episode_scores = []

        for t in range(max_t):
            action = agent.act(state, eps)
            obs_ram, obs_objects, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, obs_objects, done) # use obs_objects instead of obs_ram
            state = np.array(obs_objects, dtype=np.uint32)
            score += reward

            #records for generating replay
            episode_states.append(obs_ram)
            episode_rewards.append(reward)
            episode_scores.append(score)

            if done: break
        
        print(f"agent#{i_episode} scored {score}")
        record_history["states"] = episode_states
        record_history["rewards"] = episode_rewards
        record_history["scores"] = episode_scores
        with open(f'Replay_object_xy/ep{i_episode}_score{score}.pkl', 'wb') as f:
            pickle.dump(record_history, f)

        scores.append(score)
        steps.append(t+1)
        eps = max(eps * eps_decay, eps_end)

# pickle로 저장
data = {
    'steps': steps,
    'scores': scores
}

with open('Replay_object_xy/trajectory.pkl', 'wb') as f:
    pickle.dump(data, f)