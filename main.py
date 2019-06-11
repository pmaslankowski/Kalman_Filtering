import numpy as np
import pygame
from matplotlib import pyplot as plt

from kalman_filter_localizer import KalmanFilterLocalizer
from noisy_localizer import NoisyLocalizer
from robot import Robot

SCREEN_HEIGHT = 960
SCREEN_WIDTH = 1100
FPS = 30
TIME_PER_FRAME = 1000 // FPS


class Plotter(object):

    def __init__(self):
        self.times = [0]
        self.noisy_errors = [0]
        self.kalman_errors = [0]

    def add(self, dt, noisy_error, kalman_error):
        self.times.append(self.times[-1] + dt)
        self.noisy_errors.append(noisy_error)
        self.kalman_errors.append(kalman_error)

    def plot(self):
        plt.title('Localization error for noisy gps localizer and kalman filter')
        plt.plot(self.times, self.noisy_errors, label='noisy error')
        plt.plot(self.times, self.kalman_errors, label='kalman errors')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_HEIGHT, SCREEN_WIDTH))
    done = False

    robot = Robot(SCREEN_WIDTH, SCREEN_HEIGHT)
    noisy_localizer = NoisyLocalizer(robot, 50.)
    kalman_filter_localizer = KalmanFilterLocalizer(noisy_localizer)
    plotter = Plotter()

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
        elif pressed[pygame.K_p]:
            plotter.plot()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        pygame.time.wait(TIME_PER_FRAME)

        robot.update(TIME_PER_FRAME / 1000.)
        noisy_localizer.update()
        kalman_filter_localizer.update(TIME_PER_FRAME / 1000)

        noisy_error = np.sqrt(np.sum((noisy_localizer.location - robot.state[:2]) ** 2))
        kalman_error = np.sqrt(np.sum((kalman_filter_localizer.state[:2] - robot.state[:2]) ** 2))
        plotter.add(TIME_PER_FRAME / 1000, noisy_error, kalman_error)
        print('noisy localizer error = ', noisy_error)
        print('kalman localizer error = ', kalman_error)


        screen.fill((0, 0, 0))
        robot.draw(screen)
        noisy_localizer.draw(screen)
        kalman_filter_localizer.draw(screen)
        pygame.display.flip()