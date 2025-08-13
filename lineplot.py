import pickle
import matplotlib.pyplot as plt
import glob
import os
import re

pkl_files = glob.glob('v1/history*_score-*.pkl')

pkl_files.sort(key=lambda x: int(re.search(r'history(\d+)_', x).group(1)))

scores = []
episodes = []

for file in pkl_files:
    with open(file, 'rb') as f:
        pickle.load(f) 
    ep = int(re.search(r'history(\d+)_', file).group(1))
    episodes.append(ep)
    score = float(file.split('-')[-1].replace('.pkl', ''))
    scores.append(score)

plt.figure(figsize=(10, 5))
plt.plot(episodes, scores, marker='o', color='green')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Score Progression over Episodes')
plt.grid(True)
plt.tight_layout()
plt.show()
