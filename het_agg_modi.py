#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:43:26 2020

@author: litianle
"""

"""
Created on Wed Dec 30 20:58:33 2020

@author: litianle
"""
#dimension unify
#data loader
#build a file of
#batch


#random walk
"""
(p/u)id: (p/u)id (p/u)id (p/u)id (p/u)id (p/u)id...
"""
#data file
#data loader

"""
node_type id label embedding vector
neigbors
post neighbor:
id embed
user neighbor
id embed
"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import time
import copy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_post, neighbor_list_user, label = None):
        self.node_type = node_type
        self.node_id = node_id 
        self.embed = embed 
        self.label = label #only post node, user node = default = None
        self.neighbors_user = neighbor_list_user #[(id, embedding)]
        self.neighbors_post = neighbor_list_post
        
   
        
    
class Het_GNN(nn.Module):
    #features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim, batch_size, features, num_layers=1, rnn_type='LSTM', embed_d = 128):
        super(Het_GNN, self).__init__()
        self.input_dim = input_dim
        self.ini_hidden_dim = ini_hidden_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_d = embed_d
        self.features = features 
        self.act = nn.LeakyReLU()
        # Define the initial linear hidden layer
        self.init_linear_text = nn.Linear(self.input_dim[0], self.ini_hidden_dim[0])
        self.init_linear_image = nn.Linear(self.input_dim[1], self.ini_hidden_dim[1])
        self.init_linear_other = nn.Linear(self.input_dim[2], self.ini_hidden_dim[2])
        # Define the LSTM layer
        self.lstm_text = eval('nn.' + rnn_type)(self.input_dim[0], self.hidden_dim, self.num_layers, batch_first=True,
                                                bidirectional=True)
        self.lstm_image = eval('nn.' + rnn_type)(self.input_dim[1], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True)
        self.lstm_other = eval('nn.' + rnn_type)(self.input_dim[2], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True)
      
    def Bi_RNN(self, input, node_type):
        # Forward pass through initial hidden layer
        if node_type == "post":
            linear_input_text = self.init_linear_text(input[0])
            linear_input_image = self.init_linear_image(input[1])
            lstm_out_text, self.hidden_text = self.lstm_text(linear_input_text)
            lstm_out_image, self.hidden_image = self.lstm_image(linear_input_image)
            concate = torch.Tensor([lstm_out_text, lstm_out_image])
        if node_type == "user":
            linear_input_text = self.init_linear_text(input[0])
            linear_input_other = self.init_linear_other(input[1])
            lstm_out_text, self.hidden_text = self.lstm_text(linear_input_text)
            lstm_out_other, self.hidden_other = self.lstm_other(linear_input_other)
            concate = torch.Tensor([lstm_out_text, lstm_out_other])

        # mean pooling all the states
        mean_pooling = torch.mean(concate, 0)
        return mean_pooling
    
    
    #features: list of [(id, embedings)]
    def SameType_Agg_Bi_RNN(self, features, node_type, input_dim, ini_hidden_dim, hidden_dim, batch_size, output_dim, num_layers=1, rnn_type='LSTM'):
        extract_features = [x[1] for x in features]
        content_embedings = self.Bi_RNN(extract_features, node_type)
        init_linear = nn.Linear(input_dim, ini_hidden_dim)
        lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        linear = nn.Linear(hidden_dim * 2, output_dim)
        linear_input = init_linear(content_embedings)
        lstm_out, hidden = lstm(linear_input)
        last_state = linear(lstm_out)
        mean_pooling = torch.mean(last_state, 0)
        return mean_pooling
    
    
   
    def node_het_agg(self, het_node, u_input_dim, u_hidden_dim, u_batch_size, u_output_dim, u_num_layers, u_rnn_type,
                 p_input_dim, p_hidden_dim, p_batch_size, p_output_dim, p_num_layers, p_rnn_type): #heterogeneous neighbor aggregation

        #attention module
        c_agg_batch = self.Bi_RNN([het_node.embed], het_node.node_type)
        u_agg_batch = self.SameType_Agg_Bi_RNN(het_node.neighbors_user, "user", u_input_dim, u_hidden_dim, u_batch_size, u_output_dim, u_num_layers, u_rnn_type)
        p_agg_batch = self.SameType_Agg_Bi_RNN(het_node.neighbors_post, "post", p_input_dim, p_hidden_dim, p_batch_size, p_output_dim, p_num_layers, p_rnn_type)

        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        u_agg_batch_2 = torch.cat((c_agg_batch, u_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

        #compute weights
        concate_embed = torch.cat((c_agg_batch_2, u_agg_batch_2, p_agg_batch_2), 1).view(len(c_agg_batch), 4, self.embed_d * 2)
        if het_node.node_type == "user":
            atten_w = self.act(torch.bmm(concate_embed, self.u_neigh_att.unsqueeze(0).expand(len(c_agg_batch),*self.u_neigh_att.size())))
        else:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),*self.p_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)

        #weighted combination
        concate_embed = torch.cat((c_agg_batch, u_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), 4, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

        return weight_agg_batch
    
    def cross_entropy_loss(c_embed_batch, embed_d, outemb_d, true_label):

    	batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]
        # make c_embed 3D tensor. Batch_size * 1 * embed_d
        # c_embed[0] = 1 * embed_d
    	c_embed = c_embed_batch.view(batch_size, 1, embed_d)
    	c_embed_out = nn.Linear(embed_d, outemb_d)
    	predictions = F.logsigmoid(c_embed_out) #log(1/(1+exp(-x)))    sigmoid = 1/(1+exp(-x))
    	#binary cross entropy loss
    	loss = nn.BCELoss()
    	loss_mean = loss(predictions, true_label)
    	return loss_sum.mean()
        
    