import pygame
import random
import sys
import time

from src.Dino import Dino
from src.Obstacle import Obstacle
from src.constants import *

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simple Dino Game")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)


# --- Game loop ---
def main():
    dino = Dino()
    obstacles = []
    game_started = False
    speed = 6
    spawn_timer = 0
    start_time = 0
    running = True

    while running:
        screen.fill((255, 255, 255))
        pygame.draw.line(screen, (100, 100, 100), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if not game_started and event.key in [pygame.K_UP, pygame.K_SPACE]:
                    game_started = True
                    start_time = time.time()
                if game_started:
                    if event.key in [pygame.K_UP, pygame.K_SPACE]:
                        dino.jump()
                    elif event.key == pygame.K_DOWN:
                        dino.duck()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    dino.unduck()

        if not game_started:
            msg = font.render("Press UP or SPACE to start/jump", True, (50, 50, 50))
            screen.blit(msg, (SCREEN_WIDTH // 3, SCREEN_HEIGHT // 2))
            pygame.display.flip()
            clock.tick(30)
            continue

        # --- Update Dino ---
        dino.update()

        # --- Obstacle spawning ---
        spawn_timer += 1
        if spawn_timer > random.randint(60, 90):
            kind = random.choice(["tree", "bird"])
            obstacles.append(Obstacle(SCREEN_WIDTH, kind))
            spawn_timer = 0

        # --- Update obstacles ---
        for obs in obstacles:
            obs.update(speed)
        obstacles = [obs for obs in obstacles if not obs.is_off_screen()]

        # --- Collision check ---
        dino_rect = pygame.Rect(dino.x, dino.y, dino.width, dino.height)
        game_over = False
        for obs in obstacles:
            obs_rect = pygame.Rect(obs.x, obs.y, obs.width, obs.height)
            if dino_rect.colliderect(obs_rect):
                game_over = True
                break

        # --- Draw everything ---
        dino.draw(screen)
        for obs in obstacles:
            obs.draw(screen)

        # --- Score (time-based) ---
        score = int(time.time() - start_time)
        score_surface = font.render(f"Score: {score}", True, (10, 10, 10))
        screen.blit(score_surface, (10, 10))

        if game_over:
            msg = font.render("Game Over! Press ESC to quit.", True, (200, 50, 50))
            screen.blit(msg, (SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2))
            pygame.display.flip()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                clock.tick(30)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
