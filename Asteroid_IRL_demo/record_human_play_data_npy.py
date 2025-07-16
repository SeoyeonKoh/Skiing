import numpy as np
import keyboard
import time
import os
from ocatari.core import OCAtari
import ocatari.ram.asteroids as OCA_asteroids
import gc

### 구간별로 쪼개서 저장한 npy 파일들을 나중에 하나로 합쳐줘야함

# 설정
save_dir = "./human_play_data/subject05"
os.makedirs(save_dir, exist_ok=True)
BUFFER_SAVE_FREQ = 500  # 프레임당 저장 주기

# 환경 초기화
env = OCAtari(env_name="Asteroids-ramNoFrameskip-v4", mode="vision", obs_mode="ori", render_mode="human", buffer_window_size=1)
objects = OCA_asteroids._init_objects_ram(hud=False)

# 기록 초기화
rams_record, objects_record, actions = [], [], []
rewards, steps_in_game = [], []

# 저장 함수 정의
def save_buffered_data(index):
    np.save(os.path.join(save_dir, f"rams_part{index}.npy"), np.array(rams_record))
    np.save(os.path.join(save_dir, f"objects_part{index}.npy"), np.array(objects_record))
    np.save(os.path.join(save_dir, f"actions_part{index}.npy"), np.array(actions))
    np.save(os.path.join(save_dir, f"rewards_part{index}.npy"), np.array(rewards))
    np.save(os.path.join(save_dir, f"steps_part{index}.npy"), np.array(steps_in_game))

# 시작
start_time = time.time()
game_frame_counter = 0
buffer_index = 0

print("▶️ 게임 시작. 10분 동안 플레이하거나 Ctrl+C로 수동 종료 가능.")

try:
    prev_lives = None

    while True:
        obs, info = env.reset()
        terminated, truncated = False, False
        step_in_episode = 0
        prev_lives = info.get("lives", 3)

        while not terminated and not truncated:
            if game_frame_counter >= 36000:
                raise TimeoutError("🛑 10분 경과 - 자동 종료")

            # 키보드 입력 처리
            action = 0
            if keyboard.is_pressed('space'):
                action = 1
            if keyboard.is_pressed('up'):
                action = 2
            if keyboard.is_pressed('down'):
                action = 5
            if keyboard.is_pressed('left'):
                action = 4
            if keyboard.is_pressed('right'):
                action = 3

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()

            # 라이프 감소 시 스텝 리셋
            current_lives = info.get("lives", prev_lives)
            if current_lives < prev_lives:
                print(f"💀 라이프 감소 감지! 스텝 카운터 리셋 (남은 라이프: {current_lives})")
                step_in_episode = 0
            prev_lives = current_lives

            # 기록 저장
            rams_record.append(ram)
            actions.append(action)
            rewards.append(reward)
            steps_in_game.append(step_in_episode)

            OCA_asteroids._detect_objects_ram(objects, ram, hud=False)
            temp_objects = []
            temp_objects.extend(objects[0]._nsrepr)
            for i in range(32):
                temp_objects.extend(objects[i + 1].xy + objects[i + 1].wh)
            objects_record.append(temp_objects)

            # 렌더링 (성능유지 위해 간헐적 수행)
            if game_frame_counter % 5 == 0:
                env.render()

            step_in_episode += 1
            game_frame_counter += 1

            # 버퍼 저장
            if game_frame_counter % BUFFER_SAVE_FREQ == 0:
                print(f"💾 {game_frame_counter}프레임 - 중간 저장 실행")
                save_buffered_data(buffer_index)
                buffer_index += 1
                rams_record.clear()
                objects_record.clear()
                actions.clear()
                rewards.clear()
                steps_in_game.clear()
                gc.collect()

except KeyboardInterrupt:
    print("\n🧍 사용자에 의해 수동 종료되었습니다.")
except TimeoutError as e:
    print(f"\n{e}")
finally:
    env.close()
    print(f"\n💾 총 {game_frame_counter}프레임을 기록 중... 잔여 데이터 저장 시작.")

    if rams_record:
        save_buffered_data(buffer_index)

    print(f"✅ 저장 완료: '{save_dir}' 폴더에 기록됨.")