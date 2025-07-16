OCAtari: Atari 게임들을 gymnasium 환경에서 실행하면서, 게임상에 존재하는 object들의 정보에 접근할 수 있게 해주는 라이브러리. 실제로 게임이 굴러갈 때 모든 데이터는 RAM이라는 방식으로 관리됨. 각 프레임별로 RAM state는 리스트 형식으로 정의되며, 각 자리마다 특정 정보가 저장되어 있음. 오브젝트들의 x좌표, y좌표, 점수, 목숨 등등. 그러나 어떤 자리에는 두가지 이상의 정보가 섞여 있는 경우도 있어서 직접 그 규칙을 알아내기는 번거로움이 있음. OCAtari 라이브러리에 있는 _detect_objects_ram, .nsrepr 등의 함수나 property들을 사용하면, 특정 RAM state를 넣어서 내가 원하는 정보를 뽑아낼 수 있음.

DQN_train_RAM : observation space를 RAM state로 하여 DQN 학습하는 코드. DQN 관련해서는 간소화된 라이브러리를 써서 실행이 되는지만 빠르게 확인했었음.

DQN_train_object_xywh.py: RAM state를 바로 사용하지 않고, 플레이어+운석+총알들의 위치와 크기에 대한 벡터를 observation space로 하여 DQN 학습하는 코드.

DQN_plot_trajectory: 위의 학습코드의 각 episode마다의 score를 그래프로 시각화하는 코드.

DQN_replay_mp4: 사람이나 AI의 프레임별 플레이 기록 (states, rewards로 구성) 을 불러와서, 리플레이 mp4영상을 만들어내는 코드

record_human_play_data_npy: 사람이 직접 게임 플레이하면서 지정한 길이만큼 리플레이 데이터를 기록하는 코드. 게임환경의 상태 2종류 (RAM/object), action, reward, step (가장 최근에 죽은 후로부터 몇번째 step인지) 로서 총 5종류를 기록함. 중간에 프레임저하가 심하게 일어나는 문제가 있어서 1000이나 500프레임단위로 끊어서 npy파일을 저장함. 그러나 아직도 플레이하는 중간에 프레임저하가 심할 때가 있음.