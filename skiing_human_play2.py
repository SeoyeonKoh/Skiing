from ocatari.core import OCAtari
import pygame, sys

DO_NOTHING, RIGHT, LEFT = 0, 1, 2

env = OCAtari("Skiing", mode="ram", render_mode="rgb_array", hud=True)
obs, info = env.reset()

pygame.init()
first_frame = env.render()
h, w = first_frame.shape[:2]
screen = pygame.display.set_mode((w, h))
clock = pygame.time.Clock()

print("조작: ←(LEFT), →(RIGHT), ESC 종료")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        action = LEFT
    elif keys[pygame.K_RIGHT]:
        action = RIGHT
    else:
        action = DO_NOTHING

    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()                    
    surf = pygame.surfarray.make_surface(frame.swapaxes(0,1))  
    screen.blit(surf, (0,0))
    pygame.display.flip()

    if terminated or truncated:
        obs, info = env.reset()

    clock.tick(30)  

env.close()
pygame.quit()
sys.exit()
