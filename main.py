import numpy as np
import pygame

from kalman_filter_localizer import KalmanFilterLocalizer
from noisy_localizer import NoisyLocalizer
from robot import Robot

SCREEN_HEIGHT = 960
SCREEN_WIDTH = 1100
FPS = 30
TIME_PER_FRAME = 1000 // FPS

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
    done = False

    robot = Robot(SCREEN_WIDTH, SCREEN_HEIGHT)
    noisy_localizer = NoisyLocalizer(robot, 50.)
    kalman_filter_localizer = KalmanFilterLocalizer(noisy_localizer)

    while not done:
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_UP]:
            robot.up()
            kalman_filter_localizer.up()
        elif pressed[pygame.K_DOWN]:
            robot.down()
            kalman_filter_localizer.down()
        elif pressed[pygame.K_LEFT]:
            robot.left()
            kalman_filter_localizer.left()
        elif pressed[pygame.K_RIGHT]:
            robot.right()
            kalman_filter_localizer.right()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        pygame.time.wait(TIME_PER_FRAME)

        robot.update(TIME_PER_FRAME / 1000.)
        noisy_localizer.update()
        kalman_filter_localizer.update(TIME_PER_FRAME / 1000)

        screen.fill((0, 0, 0))
        robot.draw(screen)
        noisy_localizer.draw(screen)
        kalman_filter_localizer.draw(screen)
        pygame.display.flip()