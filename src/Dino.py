import pygame
from src.constants import *


class Dino:
    def __init__(self):
        self.x = 50
        self.y = GROUND_Y - DINO_HEIGHT
        self.width = DINO_WIDTH
        self.height = DINO_HEIGHT
        self.velocity_y = 0
        self.is_jumping = False
        self.is_ducking = False

    def jump(self):
        if not self.is_jumping:
            self.velocity_y = JUMP_VELOCITY
            self.is_jumping = True

    def duck(self):
        self.is_ducking = True
        self.height = DINO_HEIGHT // 2

        if not self.is_jumping:
            self.y = GROUND_Y - self.height

        self.velocity_y += 2*GRAVITY

    def unduck(self):
        if self.is_ducking:
            self.is_ducking = False
            self.height = DINO_HEIGHT
            self.y = GROUND_Y - self.height

    def update(self):
        self.y += self.velocity_y
        if self.is_jumping:
            self.velocity_y += GRAVITY
        if self.y >= GROUND_Y - self.height:
            self.y = GROUND_Y - self.height
            self.velocity_y = 0
            self.is_jumping = False

    def draw(self, screen):
        color = (100, 100, 100)
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))
