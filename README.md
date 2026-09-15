# Skiing DQN

OCAtari/Gymnasium 기반으로 Atari `Skiing` 환경을 직접 플레이하고, RAM state를 이용해 DQN 학습 및 리플레이 영상을 만드는 실험 코드입니다.

원본 작업 브랜치: [`KAIST-PAI-lab/Atari_DQN`의 `seoyeon` 브랜치](https://github.com/KAIST-PAI-lab/Atari_DQN/tree/seoyeon)

## 프로젝트 구성

| 파일/폴더 | 설명 |
|---|---|
| `human_play2.py` | 키보드로 Skiing을 직접 플레이하는 스크립트 |
| `DQN_skiing2.py` | RAM observation 기반 DQN 학습 스크립트 |
| `mp4.py` | 학습/플레이 기록 `.pkl`을 리플레이 `.mp4`로 변환 |
| `plot.py` | episode별 score를 막대그래프로 시각화 |
| `lineplot.py` | episode별 score 변화를 선 그래프로 시각화 |
| `Asteroid_IRL_demo/` | Asteroids 환경에서 RAM/object 기반 DQN, human play data 기록, 리플레이 생성 실험 |

## 참고 자료

- DQN sample code: <https://github.com/CCS-Lab/project_highway_irl_public/blob/main/2_dqn_train_exp_iter.py>
- OCAtari repository: <https://github.com/k4ntz/OC_Atari>
- ALE Skiing documentation: <https://ale.farama.org/environments/skiing/>

## 실행 준비

Python 환경에서 아래 라이브러리가 필요합니다.

```bash
pip install numpy gymnasium ale-py ocatari pygame opencv-python pillow matplotlib
```

Atari ROM/환경 설정은 사용하는 `ale-py`, `gymnasium`, `ocatari` 버전에 따라 추가 설정이 필요할 수 있습니다.

## 실행 예시

직접 플레이:

```bash
python human_play2.py
```

리플레이 영상 생성:

```bash
python mp4.py --pkl path/to/history1_score-13634.0.pkl --out videos/replay.mp4 --fps 30
```

점수 시각화:

```bash
python plot.py
python lineplot.py
```

## 현재 확인할 점

- `DQN_skiing2.py`는 `from dqn.agent import Agent`를 사용하지만, 현재 업로드된 파일에는 `dqn/agent.py`가 없습니다. 로컬에 남아 있는 파일이 있으면 추가 업로드가 필요합니다.
- `DQN_skiing2.py`의 `n_episodes = ###` 값은 실행 전 숫자로 설정해야 합니다.
- `plot.py`, `lineplot.py`는 기본적으로 `v1/history*_score-*.pkl` 경로를 읽습니다. 실제 기록 폴더명에 맞게 수정해서 실행하세요.
- `Asteroid_IRL_demo/human_play_data/`에는 실험 데이터 `.npy` 파일이 많이 포함되어 있습니다. 코드 중심 공개용으로 정리하려면 데이터는 release, Drive, 또는 별도 storage로 옮기는 것을 권장합니다.

## 잔여 업로드 체크리스트

로컬 원본 폴더를 다시 확인할 때 아래 파일/폴더가 빠졌는지 우선 확인하면 됩니다.

- `dqn/agent.py` 또는 DQN agent 구현 파일
- 학습 결과 폴더 `v1/`, `dqn/`, `videos/`
- `requirements.txt` 또는 사용한 Python 환경 기록
- 최종 score/episode 결과 파일
