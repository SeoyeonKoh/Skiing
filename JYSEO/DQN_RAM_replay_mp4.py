import pickle
import cv2
import numpy as np
from ocatari.core import OCAtari
from PIL import Image, ImageDraw

def history_to_video(history_path, output_video_path, fps=30):
    # history 파일 불러오기
    with open(history_path, 'rb') as f:
        data = pickle.load(f)
    states = data['states']
    rewards = data['rewards']

    # OCAtari 환경 초기화
    env = OCAtari(env_name="BattleZoneNoFrameskip-v4", mode="vision", obs_mode="ori", render_mode="rgb_array", buffer_window_size=1)
    env.reset()

    # 첫 프레임에서 크기 추출, 비디오 라이터 초기화
    for j, val in enumerate(states[0]):
        env.set_ram(j, val)
    env.step(0)
    frame_array = env.get_rgb_state
    height, width, _ = frame_array.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # state, reward 하나씩 env에 반영, env의 rgb state를 이미지로 전환 후 영상에 추가
    for i, ram in enumerate(states):
        for j, val in enumerate(ram):
            env.set_ram(j, val)
        env.step(0)
        rgb = env.get_rgb_state
        img = Image.fromarray(rgb, 'RGB')

        # 화면 구석에 reward 표시하기
        draw = ImageDraw.Draw(img)
        reward_text = f"Reward: {int(rewards[i])}"
        draw.text((5, height - 15), reward_text, fill=(255, 255, 255))

        # PIL 이미지를 OpenCV 형식으로 변환 후 기록
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)

        if i % 1000 == 0:
            print(f"Processed frame {i}/{len(states)}")

    out.release()
    print(f"🎬 Video saved to: {output_video_path}")

#pkl_path = "ep1_score1030.0.pkl"
#output_video_path = "Replayvid_object_xywh_ep1_score1030.mp4"
#history_to_video(pkl_path, output_video_path, fps=30)

history_to_video(
    history_path="/content/drive/MyDrive/Colab Notebooks/dqn/history_ep0994_score21000.0.pkl",
    output_video_path="/content/drive/MyDrive/Colab Notebooks/dqn/video1/RAM_ep0994_score21000_2.mp4",
    fps=45)