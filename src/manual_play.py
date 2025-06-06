import pygame
import time

from src.dino_env import DinoEnv


def play_manually(env):
    obs = env.reset()
    done = False
    total_reward = 0
    action = 0  # Default: do nothing

    print("Controls: [UP] or [SPACE] to jump, [DOWN] to duck, release [DOWN] to unduck, [ESC] to quit.")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_SPACE]:
                    action = 1  # Jump
                elif event.key == pygame.K_DOWN:
                    action = 2  # Duck
                elif event.key == pygame.K_ESCAPE:
                    done = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    action = 0  # Stop ducking (back to running)
                elif event.key in [pygame.K_UP, pygame.K_SPACE]:
                    action = 0  # Stop jumping (let gravity do its work)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.01)  # Slow down loop for playability

    print("Game over! Your score:", env.score)
    pygame.quit()

if __name__ == "__main__":
    env = DinoEnv(render_mode=True)
    play_manually(env)