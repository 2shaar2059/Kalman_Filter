"""
HELPFUL LINKS:
	https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/ (see pdf)
	https://www.youtube.com/watch?v=1jM6msCggYY
	https://www.youtube.com/watch?v=ri8COs2TR2U
	https://machinelearningspace.com/object-tracking-python/
"""
import numpy as np


def is_symmetric(A, tol=1e-10):
    return np.linalg.norm(A - A.T, np.Inf) < tol


def isInBetween(x, y, z):
    """returns True if every element
            in x is in-between corresponding
            elements of y and z
    """
    x_is_greater_than_y = np.greater(x, y).all()
    x_is_less_than_z = np.less(x, z).all()

    x_is_greater_than_z = np.greater(x, z).all()
    x_is_less_than_y = np.less(x, y).all()

    return ((x_is_greater_than_y and x_is_less_than_z) or (x_is_greater_than_z and x_is_less_than_y))


class Kalman_Filter(object):
    def __init__(self, A, B, H, P0, Q, R, x0, dt):
        self.A = np.asmatrix(A)
        self.B = np.asmatrix(B)
        self.H = np.asmatrix(H)
        self.P = np.asmatrix(P0)
        self.Q = np.asmatrix(Q)
        self.R = np.asmatrix(R)
        self.x_forecasted = x0
        self.x_estimated = x0
        self.dt = dt

    # forecasting what the new state SHOULD be based on the current state, current input, and the model parameters (A and B matrices)
    def forecast(self, u):
        print(self.A, self.x_estimated, self.x_forecasted)
        self.x_forecasted = self.A * self.x_estimated + self.B * u
        self.P = self.A @ self.P @ self.A.T + self.Q

        # making sure the error covariance matrix is symmetric
        assert is_symmetric(self.P)

    # calculating the kalman gain and using that to update/correct the forecasted state based on the new sensor measurements
    def estimate(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x_estimated = self.x_forecasted + K * (z - self.H * self.x_forecasted)
        
        # updating P using the P.D. Joseph update equation:
        term1 = (np.eye(self.H.shape[1]) - K * self.H)
        term2 = (K @ self.R @ K.T).T
        self.P = term1 @ self.P @ term1.T + term2

        # making sure the updated estimate is in-between the prediction and the sensor reading
        assert isInBetween(self.H * self.x_estimated, self.H * self.x_forecasted, z), f"{self.H*self.x_estimated}, {self.H*self.x_forecasted}, {z}"

        # making sure the error covariance matrix is symmetric
        assert is_symmetric(self.P)
