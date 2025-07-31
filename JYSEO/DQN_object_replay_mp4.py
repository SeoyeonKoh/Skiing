import pickle
import cv2
import numpy as np
from ocatari.core import OCAtari
from PIL import Image, ImageDraw

def history_to_video(history_path, output_video_path, fps=30):
    # 1) 기록 로드
    with open(history_path, 'rb') as f:
        data = pickle.load(f)
    rams    = data['rams']      
    rewards = data['rewards']   

    # 2) 픽셀(vision) 모드 BattleZone 환경 초기화
    env = OCAtari(
        env_name="BattleZoneNoFrameskip-v4",
        mode="vision",
        obs_mode="ori",
        render_mode="rgb_array",
        buffer_window_size=1
    )
    frame, _ = env.reset()
    height, width, _ = frame.shape

    # 3) 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 4) RAM → 화면 재구성 → 프레임 저장 루프
    for i, ram in enumerate(rams):
        # raw RAM 바이트 덮어쓰기
        for idx, byte in enumerate(ram):
            env.set_ram(idx, int(byte))

        # no-op 액션으로 화면 갱신
        frame, _, _, _, _ = env.step(0)

        # 보상 텍스트 오버레이
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        draw.text((5, height-20), f"Reward: {int(rewards[i])}", fill=(255,255,255))

        # OpenCV 포맷으로 변환 후 기록
        out.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        if (i+1) % 500 == 0:
            print(f"Processed frame {i+1}/{len(rams)}")

    # 5) 마무리
    out.release()
    env.close()
    print(f"🎬 Video saved to: {output_video_path}")


# 사용 예
history_to_video(
    history_path="/content/drive/MyDrive/Colab Notebooks/dqn/object_with_replay/history_ep0975_score11000.0.pkl",
    output_video_path="/content/drive/MyDrive/Colab Notebooks/dqn/video2/object_ep0975_score11000.mp4",
    fps=45
)