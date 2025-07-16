import pickle
import matplotlib.pyplot as plt

# trajectory.pkl 파일 로드
with open('Replay_object_xy/trajectory.pkl', 'rb') as f:
    data = pickle.load(f)

steps = data['steps']
scores = data['scores']

# 시계열 그래프 그리기
plt.figure(figsize=(12, 5))

# steps 그래프
plt.subplot(1, 2, 1)
plt.plot(steps, label='Steps per episode', color='blue')
plt.title('Steps Over Time')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True)
plt.legend()

# scores 그래프
plt.subplot(1, 2, 2)
plt.plot(scores, label='Scores per episode', color='green')
plt.title('Scores Over Time')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()