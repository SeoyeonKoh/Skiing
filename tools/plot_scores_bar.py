import pickle
import matplotlib.pyplot as plt
import glob
import os

# v1 폴더 안의 history*.pkl 파일 목록 불러오기
pkl_files = glob.glob('v2/history*_score-*.pkl')

scores = []
file_names = []

for file in pkl_files:
    with open(file, 'rb') as f:
        data = pickle.load(f)
        # 파일 이름 저장
        file_names.append(os.path.basename(file))
        # 점수 추출
        score = float(file.split('-')[-1].replace('.pkl', ''))
        scores.append(score)

# 그래프 그리기
plt.figure(figsize=(10, 5))
plt.bar(range(len(scores)), scores, tick_label=file_names)
plt.xticks(rotation=90)
plt.ylabel('Score')
plt.title('Scores from history PKL files')
plt.tight_layout()
plt.show()
