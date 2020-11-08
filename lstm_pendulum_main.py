# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:21:58 2020

@author: Lenovo
"""

#%% Import Packages 
import torch
import numpy as np
import torch.optim as optim
import nn_pend_func as pf
# import pend_phy as pph
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import matplotlib.animation as animation

#%% User Params
Animate = True
batch_size = 1
num_of_epochs = 1000
input_window = 50
output_len = 450
train_size = 8000
#%% Import data
file_name = 'dataset_d_pend_single_long.npy'

trainset = pf.PendDataSetSingle(file_name, input_len = input_window, outputlen = output_len, train = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
trainSetSize = len(trainset)
dataiter = iter(trainloader)

#%% Initialize model and optimizer
input_size = 4
hidden_size = 100
num_layers = 1
num_classes = output_len

model = pf.LSTM_p(num_classes, input_size, hidden_size, num_layers, input_window,  batch_size = batch_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.cuda()
model.train()
#%% Train Model
for epoch in range(num_of_epochs):
    dataiter = iter(trainloader)
    for i in range(train_size//(batch_size * (input_window + output_len))):
        data_item, data_label = dataiter.next()
        data_x_torch = torch.zeros((batch_size, input_window, input_size))
        data_y_torch = torch.zeros((batch_size, output_len, input_size))

        for j in range(input_size):
            data_xj = data_item[:,j,:]
            data_yj = data_label[:,j,:]
                    
            data_x_torch[:,:,j] = (data_xj).view(batch_size, input_window).float()
            data_y_torch[:,:,j] = (data_yj).view(batch_size, output_len).float()
        
        outputs = model(data_x_torch.cuda())
        # myoutput = pf.recursive_predict(model, data_x1_torch.cuda(), 20)
        optimizer.zero_grad()
        
        # obtain the loss function
        loss = criterion(outputs.cuda().view(batch_size, output_len, -1), data_y_torch.cuda())
        
        loss.backward()
        
        optimizer.step()
    if epoch % 10 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

#%% predict
ind2plot = 0
file_name = 'dataset_d_pend_single_long.npy'

testset = pf.PendDataSetSingle(file_name, input_len = input_window, outputlen = output_len, train = False)
testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False)

dataiter = iter(testloader)
# sc = MinMaxScaler()

data_item, data_label = dataiter.next()
data_x_torch = torch.zeros((batch_size, input_window, input_size))
data_y_torch = torch.zeros((batch_size, output_len, input_size))

for j in range(input_size):
    data_xj = data_item[:,j,:]
    data_yj = data_label[:,j,:]

    # data_xj = sc.fit_transform(data_xj)
    # data_yj = sc.fit_transform(data_yj)
    
    data_x_torch[:,:,j] = (data_xj).view(batch_size, input_window).float()
    data_y_torch[:,:,j] = (data_yj).view(batch_size, output_len).float()

model.eval()
model_prediction = model(data_x_torch.cuda())

predicted_x_full = np.array([],dtype = object)
true_x_full = np.array([],dtype = object)

for j in range(input_size):
    data_xj = data_item[:,j,:]
    data_yj = data_label[:,j,:]
    dataX_plot = data_xj.data.numpy()
    dataY_plot = data_yj.data.numpy()
    
    My_train_predict = model_prediction[:,:,j]
    # My_train_predict = pf.recursive_predict(model, dataX_scaled.cuda(), train_size-input_window)
    my_data_predict = My_train_predict.cpu().data.numpy()
    
    out_x_true, out_y_true = pf.debatch_data(dataX_plot, dataY_plot)
    out_x_model, out_y_model = pf.debatch_data(dataX_plot, my_data_predict.reshape(batch_size,output_len))
    
    predicted_x_full = np.append(predicted_x_full, {'{}'.format(j): out_y_model})
    true_x_full = np.append(true_x_full, {'{}'.format(j): out_y_true})

    plt.figure()
    plt.plot(out_x_true, out_y_true, 'o')
    plt.plot(out_x_model, out_y_model, '.')
    plt.suptitle('Time-Series Prediction')
    plt.show()
    
#%% Animate
dt = 0.05

x1_m = predicted_x_full[0]['0']
y1_m = predicted_x_full[1]['1']
x2_m = predicted_x_full[2]['2']
y2_m = predicted_x_full[3]['3']

x1_t = true_x_full[0]['0']
y1_t = true_x_full[1]['1']
x2_t = true_x_full[2]['2']
y2_t = true_x_full[3]['3']

if Animate:
    fig = plt.figure()
    ax = fig.add_subplot(121, autoscale_on=False, xlim=(-6, 6), ylim=(-10, 6))
    ax.set_aspect('equal')
    ax.grid()
    plt.title('Model Prediction')

    line, = ax.plot([], [], 'o-r', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    
    input_temp = 'Initial few seconds \nare used As input!'
    input_text_t = ax.text(0.05, 0.8, '', transform=ax.transAxes)

    
    ax = fig.add_subplot(122, autoscale_on=False, xlim=(-6, 6), ylim=(-10, 6))
    ax.set_aspect('equal')
    ax.grid()
    plt.title('Ground truth')
    
    line_t, = ax.plot([], [], 'o-b', lw=2)
    time_template = 'time = %.1fs'
    time_text_t = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init_m():
        line.set_data([], [])
        time_text.set_text('')
        line_t.set_data([], [])
        time_text_t.set_text('')
        input_text_t.set_text('')
        
        return line, time_text, line_t, time_text_t, input_text_t
    
    
    def animate_m(i):
        thisx = [0, x1_m[i], x2_m[i]]
        thisy = [0, y1_m[i], y2_m[i]]
    
        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        
        thisx = [0, x1_t[i], x2_t[i]]
        thisy = [0, y1_t[i], y2_t[i]]
    
        line_t.set_data(thisx, thisy)
        time_text_t.set_text(time_template % (i*dt))
        
        if i < input_window:
            input_text_t.set_text(input_temp)
        else:
            input_text_t.set_text('model output')

        return line, time_text, line_t, time_text_t, input_text_t

    ani = animation.FuncAnimation(fig, animate_m, range(1, len(y1_m)),
                              interval=dt*1000, blit=True, init_func=init_m)
    
        
    plt.show()
    # ani_t.save('anim.gif', dpi=80, writer='imagemagick')