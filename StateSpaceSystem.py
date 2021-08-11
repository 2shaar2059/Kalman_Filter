import numpy as np


class LinearSystem(object):
    def __init__(self, A, B, C, Q, R, x0, dt):
        self.A = np.asmatrix(A)
        self.B = np.asmatrix(B)
        self.C = np.asmatrix(C)
        self.Q = np.asmatrix(Q)
        self.R = np.asmatrix(R)

        self.x = x0
        self.dt = dt

    def update(self, u):  #finding the new state based on the current state, current input, the model parameters (A and B matrices) and noise
        self.x = self.A * self.x + self.B * u + np.random.multivariate_normal(np.zeros(self.x.shape[0]), self.Q).T

    def getCurrentMeasurement(self):
        if np.shape(self.C)[0] == 1:
            return (self.C * self.x + np.random.normal(0.0, self.R)).item(0)
        else:
            return self.C * self.x + np.random.multivariate_normal(np.zeros(self.C.shape[0]), self.R).T
