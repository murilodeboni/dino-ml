# dino_env.py

import random
import time
import pygame

from src.constants import *
from src.Dino import Dino
from src.Obstacle import Obstacle


class DinoEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.screen = None
        self.font = None
        self.clock = None
        self.rng = random.Random()
        self.seed_value = None
        if self.render_mode:
            assert pygame is not None, "Pygame must be installed for rendering."
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, 240))
            self.font = pygame.font.SysFont(None, 24)
            pygame.display.set_caption("Dino RL Env")
            self.clock = pygame.time.Clock()
        self.reset()

    def seed(self, seed_value=None):
        self.seed_value = seed_value
        self.rng.seed(seed_value)

    def reset(self):
        if self.seed_value is not None:
            self.rng.seed(self.seed_value)
        self.dino = Dino()
        self.obstacles = []
        self.score = 0
        self.done = False
        self.spawn_timer = 0

        # Force an obstacle to spawn nearby
        kind = self.rng.choice(["tree", "bird"])
        first_obstacle_x = self.dino.x + 300  # Or 80–150 for more jumps/ducks
        self.obstacles.append(Obstacle(first_obstacle_x, kind))
        return self._get_state()

    def step(self, action):
        self.action = action
        difficulty = max(5, 30 - self.score // 100)  # Spawns faster as score increases

        if self.spawn_timer > self.rng.randint(difficulty, difficulty + 10):
            kind = self.rng.choice(["tree", "bird"])
            self.obstacles.append(Obstacle(SCREEN_WIDTH, kind))
            self.spawn_timer = 0

        # Actions: 0 = do nothing, 1 = jump, 2 = duck
        if action == 1:
            self.dino.jump()
        elif action == 2:
            self.dino.duck()
        else:
            self.dino.unduck()

        self.dino.update()

        # Obstacles
        self.spawn_timer += 1
        if self.spawn_timer > self.rng.randint(60, 90):
            kind = self.rng.choice(["tree", "bird"])
            self.obstacles.append(Obstacle(SCREEN_WIDTH, kind))
            self.spawn_timer = 0

        for obs in self.obstacles:
            obs.update(6)
        self.obstacles = [obs for obs in self.obstacles if not obs.is_off_screen()]

        # Collision
        dino_rect = (self.dino.x, self.dino.y, self.dino.width, self.dino.height)
        for obs in self.obstacles:
            obs_rect = (obs.x, obs.y, obs.width, obs.height)
            if self._rects_collide(dino_rect, obs_rect):
                self.done = True

        self.score += 1  # Or use time

        obs = self._get_state()
        reward = 1
        if self.done:
            reward = -50
        return obs, reward, self.done, {}

    def _rects_collide(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (ax < bx + bw and ax + aw > bx and
                ay < by + bh and ay + ah > by)

    def _get_state(self):
        # Next obstacle info
        if self.obstacles:
            next_obs = min(self.obstacles, key=lambda o: o.x if o.x > self.dino.x else float('inf'))
            distance = next_obs.x - self.dino.x
            height = next_obs.height
            kind = 0 if next_obs.kind == "tree" else 1
            y_pos = next_obs.y
        else:
            distance = SCREEN_WIDTH
            height = 0
            kind = 0
            y_pos = 0
        dino_y = self.dino.y
        dino_state = int(self.dino.is_jumping) * 1 + int(self.dino.is_ducking) * 2
        return [distance, height, kind, y_pos, dino_y, dino_state]

    def render(self):
        if not self.render_mode:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        self.screen.fill((255, 255, 255))
        pygame.draw.line(self.screen, (100, 100, 100), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
        self.dino.draw(self.screen)
        for obs in self.obstacles:
            obs.draw(self.screen)
        score_surface = self.font.render(f"Score: {self.score}", True, (10, 10, 10))
        self.screen.blit(score_surface, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode and pygame is not None:
            pygame.quit()


# Example test (headless run):
if __name__ == "__main__":
    env = DinoEnv(render_mode=False)
    s = random.random()
    env.seed(s)
    obs = env.reset()
    done = False
    while not done:
        action = random.choice([0, 1, 2])
        obs, reward, done, info = env.step(action)
        # env.render() # only if rendering
    env.close()
    print("Score:", env.score, s)
