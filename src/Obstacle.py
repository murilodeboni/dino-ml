import pygame
from src.constants import *
import random


class Obstacle:
    def __init__(self, x, kind):
        self.x = x
        self.kind = kind  # "tree" or "bird"
        if kind == "tree":
            self.y = GROUND_Y - OBSTACLE_HEIGHT
            self.width = OBSTACLE_WIDTH
            self.height = OBSTACLE_HEIGHT
        else:  # bird
            self.y = GROUND_Y - BIRD_HEIGHT - random.choice([20, 40])
            self.width = OBSTACLE_WIDTH
            self.height = BIRD_HEIGHT

    def update(self, speed):
        self.x -= speed

    def draw(self, screen):
        if self.kind == "tree":
            color = (34, 177, 76)
        else:
            color = (0, 162, 232)
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))

    def is_off_screen(self):
        return self.x + self.width < 0