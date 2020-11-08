# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:20:35 2020

@author: Eli Ovits
"""
#%% Import packages 
import numpy as np
import matplotlib.pyplot as plt
import sys
from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate
import matplotlib.animation as animation
import time
#%% User Params
save_data = True
dataset_size = 1
plot_flag = True

dt = 0.05
num_of_frames = 10000
t_end = num_of_frames*dt
#%% Create pendulum classes and functions
class DoublePend():
    def __init__(self, G, L1, L2, M1, M2):
        self.G = G
        self.L1 = L1
        self.L2 = L2
        self.M1 = M1
        self.M2 = M2
     
    @classmethod
    def random_pend(cls):
        G = np.abs((np.random.randn()*1.5+9.8))
        L1 = np.random.rand()*5
        L2 = np.random.rand()*5
        M1 = np.random.rand()*5
        M2 = np.random.rand()*5
        return cls(G, L1, L2, M1, M2)

    def get_pend_params(self):
        return self.G, self.L1, self.L2, self.M1, self.M2
    
def derivs(state, t):
    G, L1, L2, M1, M2 = state[4:]
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1+M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2/L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1+M2) * G * sin(state[0]) * cos(delta)
                - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1+M2) * G * sin(state[2]))
               / den2)

    return dydx

t = np.linspace(0, t_end, num_of_frames)

#%% simulate pendulum
# pendulum params    
# G = 12 # acceleration due to gravity, in m/s^2
# L1 = 1.5  # length of pendulum 1 in m
# L2 = 1.0  # length of pendulum 2 in m
# M1 = 1.0  # mass of pendulum 1 in kg
# M2 = 1.0  # mass of pendulum 2 in kg

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
dataset = np.array([],dtype = object)
tic = time.time()
for i in range(dataset_size):

    th1 = np.random.rand()*360-180
    w1 =  np.random.rand()*20-10
    th2 = np.random.rand()*360-180
    w2 = np.random.rand()*20-10
    
    initial_cond = th1, w1, th2, w2
    # initial state
    # pend = double_pend(G, L1, L2, M1, M2)
    pend = DoublePend.random_pend()
    
    state = np.radians([th1, w1, th2, w2])
    state = np.append(state, pend.get_pend_params())
    
    # integrate ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)
    
    L1 = pend.get_pend_params()[1]
    L2 = pend.get_pend_params()[2]
    
    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])
    
    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1
    dataset = np.append(dataset, {'pendulum_data': pend.get_pend_params(), 
                                  'location':(x1,y1,x2,y2), 
                                  'initial_cond': initial_cond})
    
    # if i % (dataset_size//10) == 0:
    #     print('{:.2f}% done'.format(i/dataset_size*100))

toc = time.time()

print('Finished generating data. Total time is {:.2f} sec'.format(toc-tic))

if save_data:
    print('Saving to file...')
    np.save('dataset_d_pend_single_long_2', dataset)
#%% plot (animation)
ind2plot = 0
G, L1, L2, M1, M2 = dataset[ind2plot]['pendulum_data']
x1,y1,x2,y2 = dataset[ind2plot]['location']

if plot_flag:
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-3, 3), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-r', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    pend_params_text = 'dataset sample #{}\nG = {:.2f}, L1 = {:.2f}, \nL2 = {:.2f}, M1 = {:.2f}, M2 = {:.2f}'.format(ind2plot,G, L1, L2, M1, M2)
    ax.text(0.05, 0.7, pend_params_text, transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    
    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
    
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                              interval=dt*1000, blit=True, init_func=init)
    plt.show()
