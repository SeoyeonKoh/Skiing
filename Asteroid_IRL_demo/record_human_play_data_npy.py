import numpy as np
import time
import os
from ocatari.core import OCAtari
import ocatari.ram.asteroids as OCA_asteroids
import gc
from pynput import keyboard

# 설정
save_dir = "./human_play_data/subject06"
os.makedirs(save_dir, exist_ok=True)
BUFFER_SAVE_FREQ = 500  # 프레임당 저장 주기

# 환경 초기화
env = OCAtari(env_name="ALE/Tennis-v5", mode="vision", obs_mode="ori", render_mode="human", buffer_window_size=1)
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

# 키 상태 저장용 변수
key_states = set()

# 키 입력 핸들러
def on_press(key):
    try:
        if hasattr(key, 'char') and key.char:
            key_states.add(key.char)
        elif hasattr(key, 'name'):
            key_states.add(key.name)
    except:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char:
            key_states.discard(key.char)
        elif hasattr(key, 'name'):
            key_states.discard(key.name)
    except:
        pass

# 리스너 시작
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

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
            if 'space' in key_states:
                if 'up' in key_states and 'right' in key_states:
                    action = 14  # UPRIGHTFIRE
                elif 'up' in key_states and 'left' in key_states:
                    action = 15  # UPLEFTFIRE
                elif 'down' in key_states and 'right' in key_states:
                    action = 16  # DOWNRIGHTFIRE
                elif 'down' in key_states and 'left' in key_states:
                    action = 17  # DOWNLEFTFIRE
                elif 'up' in key_states:
                    action = 10  # UPFIRE
                elif 'down' in key_states:
                    action = 13  # DOWNFIRE
                elif 'left' in key_states:
                    action = 12  # LEFTFIRE
                elif 'right' in key_states:
                    action = 11  # RIGHTFIRE
                else:
                    action = 1   # FIRE only
            else:
                if 'up' in key_states and 'right' in key_states:
                    action = 6  # UPRIGHT
                elif 'up' in key_states and 'left' in key_states:
                    action = 7  # UPLEFT
                elif 'down' in key_states and 'right' in key_states:
                    action = 8  # DOWNRIGHT
                elif 'down' in key_states and 'left' in key_states:
                    action = 9  # DOWNLEFT
                elif 'up' in key_states:
                    action = 2  # UP
                elif 'down' in key_states:
                    action = 5  # DOWN
                elif 'left' in key_states:
                    action = 4  # LEFT
                elif 'right' in key_states:
                    action = 3  # RIGHT
                
                

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
    listener.stop()
    print(f"\n💾 총 {game_frame_counter}프레임을 기록 중... 잔여 데이터 저장 시작.")

    if rams_record:
        save_buffered_data(buffer_index)

    print(f"✅ 저장 완료: '{save_dir}' 폴더에 기록됨.")
