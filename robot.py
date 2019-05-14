import pygame
import numpy as np


class Robot(object):

    def __init__(self, screen_width, screen_height):
        #x, y, vx, vy
        self.state = np.array([0., 0., 0., 0.]).reshape(-1, 1)
        self.accel = np.array([0., 0.]).reshape(-1, 1)
        self.radius = 5
        self.color = (0, 128, 255)
        self.accel_boost = 130.
        self.accel_sigma = 20.
        self.screen_width = screen_width
        self.screen_height = screen_height

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.state[0], self.state[1]), self.radius)

    def up(self):
        self.accel = np.array([0., -self.accel_boost]).reshape(-1, 1)
        self.accel = np.random.normal(self.accel, self.accel_sigma)

    def down(self):
        self.accel = np.array([0., self.accel_boost]).reshape(-1, 1)
        self.accel = np.random.normal(self.accel, self.accel_sigma)

    def left(self):
        self.accel = np.array([-self.accel_boost, 0.]).reshape(-1, 1)
        self.accel = np.random.normal(self.accel, self.accel_sigma)

    def right(self):
        self.accel = np.array([self.accel_boost, 0.]).reshape(-1, 1)
        self.accel = np.random.normal(self.accel, self.accel_sigma)

    def update(self, dt):
        F = np.array([[1., 0., dt, 0.],
                      [0., 1., 0., dt],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        B = np.array([[0.5 * dt ** 2, 0.],
                      [0., 0.5 * dt ** 2],
                      [dt, 0.],
                      [0., dt]])
        self.state = F.dot(self.state) + B.dot(self.accel)
        self.accel = np.array([0., 0.]).reshape(-1, 1)
        if self.state[0] < 0 or self.state[0] > self.screen_width:
            self.state[2] *= -1
        if self.state[1] < 0 or self.state[1] > self.screen_height:
            self.state[3] *= -1

        print('robot state:', self.state)