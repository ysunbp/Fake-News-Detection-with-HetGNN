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
import torch.optim as optim





class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_post = [], neighbor_list_user = [], label = None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label #only post node, user node = default = None
        self.neighbors_user = neighbor_list_user #[(id)]
        self.neighbors_post = neighbor_list_post

def data_loader(pathway = 'F:/post_nodes/', node_type = "post"):
    if node_type == "post":
        post_node = []
        post_id = []
        post_label = []
        post_embed = []
        post_p_neigh = []
        post_u_neigh = []
        for i in range(19):
            print(i)
            batch = str(i)
            f = open(pathway + "batch_" + batch + '.txt')
            print(pathway + "batch_" + batch + '.txt')
            Lines = f.readlines() 
            for j in range(len(Lines)):
                if j % 5 == 0:
                    _, id_, label = Lines[j].split()
                    post_id.append(int(id_))
                    post_label.append(int(label))
                    embed = []
                if j % 5 == 1 or j % 5 == 2:
                    embed.append(list(map(float,Lines[j].split())))
                if j % 5 == 2:
                    post_embed.append(embed)
                if j % 5 == 3:
                    post_p_neigh.append(list(map(int,Lines[j].split())))
                if j % 5 == 4:
                    post_u_neigh.append(list(map(int,Lines[j].split())))
            f.close()
        for i in range(len(post_id)):
            node = Het_Node(node_type = "post", node_id = post_id[i], embed = post_embed[i], neighbor_list_post = post_p_neigh[i], neighbor_list_user = post_u_neigh[i], label = post_label[i])
            post_node.append(node)
        return post_node
    
    else:
        user_node = []
        user_id = []
        user_embed = []
        f = open(pathway + 'user_nodes.txt')
        Lines = f.readlines() 
        for j in range(len(Lines)):
            if j % 3 == 0:
                id_ = Lines[j].split()
                #print(id_)
                user_id.append(int(id_[0]))
                embed = []
            if j % 3 == 1 or j % 3 == 2:
                embed.append(list(map(float,Lines[j].split())))
            if j % 3 == 2:
                user_embed.append(embed)
        f.close()
        for i in range(len(user_id)):
            node = Het_Node(node_type = "user", node_id = user_id[i], embed = user_embed[i])
            user_node.append(node) 
        return user_node

post_nodes = data_loader(pathway='F:/FYP_data/post_nodes/', node_type="post")
user_nodes = data_loader(pathway='F:/FYP_data/user_nodes/', node_type="user")
post_emb_dict = {}
user_emb_dict = {}
#content_dict = {}
for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb

#print(post_nodes[5].emb)



class Het_GNN(nn.Module):
    #features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim, batch_size, content_dict={}, num_layers=1, rnn_type='LSTM', embed_d = 2000):
        super(Het_GNN, self).__init__()
        self.input_dim = input_dim
        self.ini_hidden_dim = ini_hidden_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_d = embed_d
        #self.features = features
        self.content_dict = content_dict
        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim = 1)
        self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        # Define the initial linear hidden layer
        self.init_linear_text = nn.Linear(self.input_dim[0], self.ini_hidden_dim[0])
        self.init_linear_image = nn.Linear(self.input_dim[1], self.ini_hidden_dim[1])
        self.init_linear_other = nn.Linear(self.input_dim[2], self.ini_hidden_dim[2])
        # Define the LSTM layer
        self.lstm_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers, batch_first=True,
                                                bidirectional=True)
        self.lstm_image = eval('nn.' + rnn_type)(self.ini_hidden_dim[1], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True)
        self.lstm_other = eval('nn.' + rnn_type)(self.ini_hidden_dim[2], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True)

      
    def Bi_RNN(self, neighbor_id, node_type, post_emb_dict, user_emb_dict):
        # Forward pass through initial hidden layer
        input_a = []
        input_b = []
        new_id = []
        if node_type == "post":
            for i in neighbor_id:
                if ("post", i) not in self.content_dict:
                    input_a.append(post_emb_dict[i][0])
                    input_b.append(post_emb_dict[i][1])
                    new_id.append(i)
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_a)
            linear_input_image = self.init_linear_image(input_b)
            #print(linear_input_text.shape)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0],1,linear_input_text.shape[1])
            linear_input_image = linear_input_image.view(linear_input_image.shape[0],1,linear_input_image.shape[1])
            lstm_out_text, self.hidden_text = self.lstm_text(linear_input_text)
            lstm_out_image, self.hidden_image = self.lstm_image(linear_input_image)
            #print('lstm_out_image',lstm_out_image.shape)
            #print('lstm_out_text',lstm_out_text.shape)
            concate = torch.cat((lstm_out_text, lstm_out_image), 1)
            #print('concate',concate.shape)
        if node_type == "user":
            for i in neighbor_id:
                if ("user", i) not in self.content_dict:
                    input_a.append(user_emb_dict[i][0])
                    input_b.append(user_emb_dict[i][1])
                    new_id.append(i)
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_b)
            linear_input_other = self.init_linear_other(input_a)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            lstm_out_text, self.hidden_text = self.lstm_text(linear_input_text)
            lstm_out_other, self.hidden_other = self.lstm_other(linear_input_other)
            #print('lstm_out_other', lstm_out_other.shape)
            #print('lstm_out_text', lstm_out_text.shape)
            concate = torch.cat((lstm_out_text, lstm_out_other), 1)
            #print('concate', concate.shape)

        # mean pooling all the states
        mean_pooling = torch.mean(concate, 1)
        #print('mean_pooling',mean_pooling.shape)
        for i in neighbor_id:
            if ("post", i) in self.content_dict:
                mean_pooling = torch.cat(mean_pooling, self.content_dict[i], dim=0)
        for i in range(len(new_id)):
            self.content_dict[i] = mean_pooling[i]
        return mean_pooling
    
    
    #features: list of [(id)]
    def SameType_Agg_Bi_RNN(self, neighbor_id, node_type, input_dim, hidden_dim, ini_hidden_dim, batch_size, output_dim, num_layers=1, rnn_type='LSTM'):
        content_embedings = self.Bi_RNN(neighbor_id, node_type, post_emb_dict, user_emb_dict)
        init_linear = nn.Linear(input_dim, ini_hidden_dim)
        lstm = eval('nn.' + rnn_type)(ini_hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        linear = nn.Linear(hidden_dim * 2, output_dim)
        linear_input = init_linear(content_embedings)
        #print(linear_input.shape)
        linear_input = linear_input.view(linear_input.shape[0],1,linear_input.shape[1])
        lstm_out, hidden = lstm(linear_input)
        last_state = linear(lstm_out)
        mean_pooling = torch.mean(last_state, 0)
        return mean_pooling
    
    
   
    def node_het_agg(self, het_node, u_input_dim, u_hidden_dim, u_ini_hidden_dim, u_batch_size, u_output_dim, u_num_layers, u_rnn_type,
                 p_input_dim, p_hidden_dim, p_ini_hidden_dim, p_batch_size, p_output_dim, p_num_layers, p_rnn_type): #heterogeneous neighbor aggregation

        #attention module
        c_agg_batch = self.Bi_RNN([het_node.node_id], het_node.node_type, post_emb_dict, user_emb_dict)
        u_agg_batch = self.SameType_Agg_Bi_RNN(het_node.neighbors_user, "user", u_input_dim, u_hidden_dim, u_ini_hidden_dim, u_batch_size, u_output_dim, u_num_layers, u_rnn_type)
        p_agg_batch = self.SameType_Agg_Bi_RNN(het_node.neighbors_post, "post", p_input_dim, p_hidden_dim, p_ini_hidden_dim, p_batch_size, p_output_dim, p_num_layers, p_rnn_type)
        #print('c_agg_batch', c_agg_batch.shape)
        #print('u_agg_batch', u_agg_batch.shape)
        #print('p_agg_batch', p_agg_batch.shape)
        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        u_agg_batch_2 = torch.cat((c_agg_batch, u_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

        #compute weights
        concate_embed = torch.cat((c_agg_batch_2, u_agg_batch_2, p_agg_batch_2), 1).view(len(c_agg_batch), 3, self.embed_d * 2)
        if het_node.node_type == "user":
            atten_w = self.act(torch.bmm(concate_embed, self.u_neigh_att.unsqueeze(0).expand(len(c_agg_batch),*self.u_neigh_att.size())))
        else:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),*self.p_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 3)

        #weighted combination
        concate_embed = torch.cat((c_agg_batch, u_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), 3, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

        return weight_agg_batch
    
    def cross_entropy_loss(self, c_embed_batch, embed_d, outemb_d, true_label):

        batch_size = 1
        # make c_embed 3D tensor. Batch_size * 1 * embed_d
        # c_embed[0] = 1 * embed_d
        c_embed = c_embed_batch.view(batch_size, 1, embed_d)
        fc = nn.Linear(embed_d, outemb_d)
        c_embed_out = fc(c_embed)
        #print('c_embed_out', c_embed_out)
        predictions = torch.sigmoid(c_embed_out) #log(1/(1+exp(-x)))    sigmoid = 1/(1+exp(-x))
        #binary cross entropy loss
        loss = nn.BCELoss()
        #print('predictions_shape',predictions.shape)
        #print('predictions', predictions)
        predictions = predictions.view(2)
        if true_label == 1:
            tensor_label = torch.FloatTensor([1,0])
        else:
            tensor_label = torch.FloatTensor([0,1])
        loss_sum = loss(predictions, tensor_label)
        return loss_sum.mean()


net = Het_GNN([300, 512, 12], [500, 500, 500], 1000, 1)
print(net)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
running_loss = 0.0
for epoch in range(10):
    print('Epoch:', epoch+1)
    for i in range(len(post_nodes)):
        print(i)
        agg = net.node_het_agg(het_node=post_nodes[i], u_input_dim=2000, u_hidden_dim=500, u_ini_hidden_dim=500, u_batch_size=1, u_output_dim=2000, u_num_layers=1, u_rnn_type='LSTM', p_input_dim=2000, p_hidden_dim=500, p_ini_hidden_dim=500, p_batch_size=1, p_output_dim=2000, p_num_layers=1, p_rnn_type='LSTM')
        loss = net.cross_entropy_loss(c_embed_batch=agg, embed_d=2000, outemb_d=2, true_label=post_nodes[i].label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
print('Finish training')

