import numpy as np
import pygame
from scipy.stats import chi2

from numpy.linalg import inv


class KalmanFilterLocalizer(object):

    def __init__(self, noisy_localizer):
        self.noisy_localizer = noisy_localizer
        self.state = np.array([0., 0., 0., 0.]).reshape(-1, 1)
        self.cov = np.full((4, 4), 50)
        self.accel = np.array([0., 0.]).reshape(-1, 1)
        self.accel_boost = 130.
        self.accel_sigma = 20.
        self.radius = 5

    def up(self):
        self.accel = np.array([0., -self.accel_boost]).reshape(-1, 1)

    def down(self):
        self.accel = np.array([0., self.accel_boost]).reshape(-1, 1)

    def left(self):
        self.accel = np.array([-self.accel_boost, 0.]).reshape(-1, 1)

    def right(self):
        self.accel = np.array([self.accel_boost, 0.]).reshape(-1, 1)

    def get_approx_location(self):
        return self.state[:, 2]

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (self.state[0], self.state[1]), self.radius)
        w, h, angle = self.cov_ellipse(self.cov[:2, :2], 0.95)
        w = max(w, 1.)
        h = max(h, 1.)
        #print('w = {}, h = {}'.format(w, h))
        surface = pygame.Surface((w, h), pygame.SRCALPHA, 32).convert_alpha()
        pygame.draw.ellipse(surface, (0, 255, 0, 100), (0, 0, w, h))
        rot_surface = pygame.transform.rotate(surface, angle)
        rcx, rcy = rot_surface.get_rect().center
        pos = (self.state[0, 0] - rcx, self.state[1, 0] - rcy)
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

    def update(self, dt):
        F = np.array([[1., 0., dt, 0.],
                      [0., 1., 0., dt],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        B = np.array([[0.5 * dt ** 2, 0.],
                      [0., 0.5 * dt ** 2],
                      [dt, 0.],
                      [0., dt]])
        Q = np.zeros((4,4))
        Q = np.array([[5., 0., 5., 0.],
                      [0., 5., 0., 5.],
                      [5., 0., 20., 0.],
                      [0., 5., 0., 20.]])
        # predict:
        x = F.dot(self.state) + B.dot(self.accel)
        P = F.dot(self.cov).dot(F.T) + Q

        # measure:
        z = self.noisy_localizer.get_approx_location()
        R = self.noisy_localizer.get_cov_matrix()

        # update:
        H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        y = z - H.dot(x)
        S = R + H.dot(P).dot(H.T)
        K = P.dot(H.T).dot(inv(S))

        self.state = x + K.dot(y)
        self.cov = (np.eye(4) - K.dot(H)).dot(P)
        print(self.cov)
        print()

        # print('kf z:', z)
        # print('kf R:', R)
        # print('kf state: ', self.state)
        # print('kf cov', self.cov)
        # print('kf gain:', K)

        self.accel = np.array([0., 0.]).reshape(-1, 1)



