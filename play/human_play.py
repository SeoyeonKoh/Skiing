from ocatari.core import OCAtari
import pygame
import sys

# 정확한 Skiing action 번호 매핑
DO_NOTHING = 0  # 'NOOP'
RIGHT = 1       # 'RIGHT'
LEFT = 2        # 'LEFT'

# 게임 환경 설정
env = OCAtari("Skiing", mode="ram", render_mode="human", hud=True)
observation, info = env.reset()
print("Skiing 게임의 action 의미:", env.get_action_meanings())

# 키보드 입력 초기화
pygame.init()
print("조작 안내:")
print("← : 왼쪽 이동")
print("→ : 오른쪽 이동")
print("ESC : 종료")

action = DO_NOTHING
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()

    # 좌우 방향 입력만 사용
    if keys[pygame.K_LEFT]:
        action = LEFT
    elif keys[pygame.K_RIGHT]:
        action = RIGHT
    else:
        action = DO_NOTHING

    print(f"현재 선택된 action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("게임 종료. 다시 시작합니다!")
        observation, info = env.reset()

    clock.tick(30)

# 게임 종료
env.close()
pygame.quit()
sys.exit()
