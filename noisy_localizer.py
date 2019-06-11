import numpy as np
import pygame
from scipy.stats import chi2


class NoisyLocalizer(object):

    def __init__(self, robot, sigma):
        self.robot = robot
        self.sigma = sigma
        self.color = (255, 255, 0)
        self.location = np.array([0., 0.]).reshape(-1, 1)
        self.radius = 5

    def get_approx_location(self):
        return self.location

    def get_cov_matrix(self):
        return np.eye(2) * self.sigma

    def update(self):
        self.location = np.random.normal(self.robot.state[:2], self.sigma)
        #print('noisy loc:', self.location)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.location, self.radius)
        w, h, angle = self.cov_ellipse(np.eye(2) * self.sigma, 0.95)
        w = max(w, 1.)
        h = max(h, 1.)
        surface = pygame.Surface((w, h), pygame.SRCALPHA, 32).convert_alpha()
        pygame.draw.ellipse(surface, (255, 255, 0, 100), (0, 0, w, h))
        rot_surface = pygame.transform.rotate(surface, angle)
        rcx, rcy = rot_surface.get_rect().center
        pos = (self.robot.state[0, 0] - rcx, self.robot.state[1, 0] - rcy)
        screen.blit(rot_surface, pos)

    @staticmethod
    def cov_ellipse(cov, q):
        """
        Parameters
        ----------
        cov : (2, 2) array
            Covariance matrix.
        q : float, optional
            Confidence level, should be in (0, 1)
        """
        q = np.asarray(q)
        r2 = chi2.ppf(q, 2)
        #print('r2 = ', r2)
        val, vec = np.linalg.eigh(cov)
        width, height = 2 * np.sqrt(val[:, None] ** 2 * r2)
        rotation = np.degrees(np.arctan2(*vec[::-1, 0]))
        return width[0], height[0], rotation