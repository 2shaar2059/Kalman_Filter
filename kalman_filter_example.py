import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import cont2discrete as c2d

from KalmanFilter import Kalman_Filter
from StateSpaceSystem import LinearSystem

np.random.seed(22)


def getU(t):
    return 0


x0 = np.matrix([0, 2]).T  #pos, vel
currTime = 0.0
dt = 0.01

A_c = np.matrix(  #actual, continuous-time A matrix
    [[0, 1], [0, 0]])
B_c = np.matrix([0, 1]).T  #actual, continuous-time B matrix
C_c = np.matrix([1, 0])  #only observing position
D_c = 0

A_d, B_d, C_d, D_d, dt = c2d((A_c, B_c, C_c, D_c), dt)  #convert A and B matrices from continuous to discrete

Q_system = [[0.02**2, 0], [0, 0.02**2]]
R_system = 0.02  #0.015
mySystem = LinearSystem(A_d, B_d, C_d, Q_system, R_system, x0, dt)

x0_mean = [0, 0]
P0 = np.matrix([[10**2, 0], [0, 10**2]])
x0_kalman_filter_guess = np.random.multivariate_normal(x0_mean, P0).T
x0_kalman_filter_guess = np.reshape(x0_kalman_filter_guess, (2, 1))
Q_filter = [[0.02**2, 0], [0, 0.02**2]]
R_filter = np.matrix([0.03**2])
kf = Kalman_Filter(A_d, B_d, C_d, P0, Q_filter, R_filter, x0_kalman_filter_guess, dt)

#initializing integration loop variables
prev_u = 0
pos_real = [mySystem.x.item(0)]
pos_sensor = [mySystem.getCurrentMeasurement()]
pos_kalman_forecasted = [kf.x_forecasted.item(0)]
pos_kalman_estimated = [kf.x_estimated.item(0)]
time = [currTime]

currTime += dt

maxTime = 1
while (currTime <= maxTime):
    mySystem.update(prev_u)  #update the system state with the previous time-step's control input
    pos_real.append(mySystem.x.item(0))  #real state without sensor noise, but with model noise

    kf.forecast(prev_u)  #create the forecasted state in the kalman filter using the previous time-step's control input (predict/forecast step)
    sensor_measurment = mySystem.getCurrentMeasurement()  #get a sensor measurement of the system's updated state
    pos_sensor.append(sensor_measurment)
    kf.estimate(sensor_measurment)  #find estimated state based on the sensor measurement that just came in (update step)

    pos_kalman_forecasted.append(kf.x_forecasted.item(0))  #current forecasted state
    pos_kalman_estimated.append(kf.x_estimated.item(0))

    curr_u = getU(currTime)  #getting the control input for the current time-step. will be used to find the updates states in the next timestep

    time.append(currTime)

    currTime += dt  #next time-step
    prev_u = curr_u

plt.title('Example of Kalman filter for tracking a moving object in 1-D')

plt.plot(time, pos_real, label='Actual Position', color='y')
plt.scatter(time, pos_sensor, label='Sensor Measurement', color='b')
plt.scatter(time, pos_kalman_forecasted, label='Kalman forecasted state', color='r')  #based on model
plt.plot(time, pos_kalman_estimated, label='Kalman updated state', color='g')  #based on model and forecasted state
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.show()
