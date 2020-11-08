# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:21:34 2020

@author: Lenovo
"""


#%% Import Packages 
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
from torch.autograd import Variable

#%% Classes
class LSTM_p(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, batch_size = 200):
        super(LSTM_p, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.fc4 = nn.Linear(hidden_size, num_classes)

        h_0 = Variable(torch.zeros(
            self.num_layers, batch_size, self.hidden_size)).cuda()
        
        c_0 = Variable(torch.zeros(
            self.num_layers, batch_size, self.hidden_size)).cuda()
        self.hidden_cell = (h_0, c_0)
    
    def forward(self, x):
        
        h_0 = Variable(torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size)).cuda()
        
        c_0 = Variable(torch.zeros(
            self.num_layers, self.batch_size, self.hidden_size)).cuda()
        self.hidden_cell = (h_0, c_0)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, self.hidden_cell)
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out1 = self.fc1(h_out).view(self.batch_size, self.num_classes, 1)
        out2 = self.fc2(h_out).view(self.batch_size, self.num_classes, 1)
        out3 = self.fc3(h_out).view(self.batch_size, self.num_classes, 1)
        out4 = self.fc4(h_out).view(self.batch_size, self.num_classes, 1)
        
        out = torch.stack((out1, out2, out3, out4), dim = 2)
        return out
    
class PendDataSet(Dataset):
    def __init__(self, file_name, input_len = 20, train_size = 80, train = True):
        self.file_name = file_name
        self.dataset = np.load(file_name, allow_pickle=True)
        self.train_size = train_size
        self.input_len = input_len
        self.train = train
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x1,y1,x2,y2 = self.dataset[idx]['location']
        if self.train:
            item = torch.tensor([x1[0:self.input_len],y1[0:self.input_len],
                                 x2[0:self.input_len],y2[0:self.input_len]])
            label = torch.tensor([x1[self.input_len:self.train_size],y1[self.input_len:self.train_size],
                     x2[self.input_len:self.train_size],y2[self.input_len:self.train_size]])

          
            return item.float(), label.float()

        else:
            item = torch.tensor([x1[self.train_size:],y1[self.train_size:],
                                 x2[self.train_size:],y2[self.train_size:]])
            return item.float()
        
class PendDataSetSingle(Dataset):
    def __init__(self, file_name, input_len = 20, outputlen = 20, train_size = 8000, train = True):
        self.file_name = file_name
        self.dataset = np.load(file_name, allow_pickle=True)
        self.train_size = train_size
        self.input_len = input_len
        self.outputlen = outputlen
        self.train = train
        
    def __len__(self):
        x1,y1,x2,y2 = self.dataset[0]['location']
        return len(x1)//(self.input_len+self.outputlen)

    def __getitem__(self, idx):
        x1,y1,x2,y2 = self.dataset[0]['location']
        
        cycle = self.input_len + self.outputlen
        start_ind_input = idx*cycle
        end_ind_input = idx*cycle + self.input_len
        start_ind_label = end_ind_input
        end_ind_label = end_ind_input + self.outputlen
        
        if self.train  == False :
            start_ind_input = start_ind_input + int(len(self)*0.8)
            end_ind_input = end_ind_input +  int(len(self)*0.8)
            start_ind_label = start_ind_label +  int(len(self)*0.8)
            end_ind_label = end_ind_label + int(len(self)*0.8)
            

        item = torch.tensor([minmaxscaler(x1[start_ind_input:end_ind_input]),minmaxscaler(y1[start_ind_input:end_ind_input]),
                             minmaxscaler(x2[start_ind_input:end_ind_input]),minmaxscaler(y2[start_ind_input:end_ind_input])])
        
        label = torch.tensor([minmaxscaler(x1[start_ind_label:end_ind_label]),minmaxscaler(y1[start_ind_label:end_ind_label]),
                              minmaxscaler(x2[start_ind_label:end_ind_label]),minmaxscaler(y2[start_ind_label:end_ind_label])])

      
        return item.float(), label.float()

    
#%% Functions
def recursive_predict(model, input_seq, output_len):
    batch_size = input_seq.shape[0]
    seq_len = input_seq.shape[1]
    num_features = input_seq.shape[2]
    output_seq = torch.zeros((batch_size, output_len, num_features))
    cur_in = input_seq
    model.hidden_cell = (torch.zeros(model.num_layers, batch_size, model.hidden_size).cuda(),
                torch.zeros(model.num_layers, batch_size, model.hidden_size).cuda())

    for i in range(output_len):
        cur_out = model(cur_in)
        output_seq[:, i, :] = cur_out
        cur_in = torch.cat((cur_in[:, 1:, :], cur_out.view(batch_size,1,1)),1)
    return output_seq

    
def debatch_data(data_input, data_output):
    batch_size = data_input.shape[0]
    input_len = data_input.shape[1]
    output_len = data_output.shape[1]
    
    seq_len = input_len+output_len

    output_y = np.zeros(batch_size * seq_len)
    output_x = np.linspace(0, batch_size * seq_len+1, batch_size * seq_len)
    for i in range(batch_size):
        output_y[i*seq_len:i*seq_len + input_len] = data_input[i, :]
        output_y[i*seq_len+ input_len:i*seq_len + seq_len] = data_output[i, :]

    return output_x, output_y

def minmaxscaler(data):
    return data#/(np.std(data))#(data-np.min(data))/(np.std(data))