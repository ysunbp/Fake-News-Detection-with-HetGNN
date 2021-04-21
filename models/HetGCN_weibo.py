#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from os import path
from torch.autograd import Variable
import torch.nn.init as init
from scipy.linalg import fractional_matrix_power
import matplotlib.pyplot as plt
import random

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_post=[], neighbor_list_user=[], label=None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label  # only post node, user node = default = None
        self.neighbors_user = neighbor_list_user  # [(id)]
        self.neighbors_post = neighbor_list_post


def data_loader(pathway='F:\\FYP_data\\normalized_post_nodes', node_type="post"):
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
                    embed.append(list(map(float, Lines[j].split())))
                if j % 5 == 2:
                    post_embed.append(embed)
                if j % 5 == 3:
                    post_p_neigh.append(list(map(int, Lines[j].split())))
                if j % 5 == 4:
                    post_u_neigh.append(list(map(int, Lines[j].split())))
            f.close()
        for i in range(len(post_id)):
            node = Het_Node(node_type="post", node_id=post_id[i], embed=post_embed[i],
                            neighbor_list_post=post_p_neigh[i], neighbor_list_user=post_u_neigh[i], label=post_label[i])
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
                user_id.append(int(id_[0]))
                embed = []
            if j % 3 == 1 or j % 3 == 2:
                embed.append(list(map(float, Lines[j].split())))
            if j % 3 == 2:
                user_embed.append(embed)
        f.close()
        for i in range(len(user_id)):
            node = Het_Node(node_type="user", node_id=user_id[i], embed=user_embed[i])
            user_node.append(node)
        return user_node


post_nodes = data_loader(pathway='5p10u/normalized_post_nodes/', node_type="post")
user_nodes = data_loader(pathway='5p10u/normalized_user_nodes/', node_type="user")
post_emb_dict = {}
user_emb_dict = {}
for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb


# In[ ]:


post_nodes_real = []
post_nodes_fake = []
for node in post_nodes:
    if node.label == 1:
        post_nodes_real.append(node)
    else:
        post_nodes_fake.append(node)
print("number of fake nodes: ", len(post_nodes_fake))
print("number of real nodes: ", len(post_nodes_real))
#fake : real = 0.9842 : 1


# In[ ]:


class Het_GNN(nn.Module):
    # features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim, batch_size,
                 u_input_dim, u_hidden_dim, u_ini_hidden_dim, u_output_dim, u_num_layers,
                 p_input_dim, p_hidden_dim, p_ini_hidden_dim, p_output_dim, p_num_layers,
                 out_embed_d, outemb_d, symmetry = False, GCN_out2=300, GCN_out1= 300, GCN_in=200, use_bias=True,
                 u_batch_size=1, p_batch_size=1, content_dict={}, num_layers=1, u_rnn_type='LSTM', p_rnn_type='LSTM',
                 rnn_type='LSTM', embed_d=200):
        super(Het_GNN, self).__init__()
        self.input_dim = input_dim
        self.ini_hidden_dim = ini_hidden_dim
        self.hidden_dim = out_embed_d // 2
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_d = out_embed_d
        self.u_input_dim = out_embed_d
        self.u_hidden_dim = u_hidden_dim
        self.u_ini_hidden_dim = u_ini_hidden_dim
        self.u_batch_size = u_batch_size
        self.u_output_dim = out_embed_d
        self.u_num_layers = u_num_layers
        self.u_rnn_type = u_rnn_type
        self.p_input_dim = out_embed_d
        self.p_hidden_dim = p_hidden_dim
        self.p_ini_hidden_dim = p_ini_hidden_dim
        self.p_batch_size = p_batch_size
        self.p_output_dim = out_embed_d
        self.p_num_layers = p_num_layers
        self.p_rnn_type = p_rnn_type
        self.out_embed_d = out_embed_d
        self.outemb_d = outemb_d
        self.GCN_in = out_embed_d
        self.GCN_out1 = GCN_out1
        #self.GCN_out1 = out_embed_d
        self.GCN_out2 = out_embed_d * 2
        self.use_bias = use_bias
        self.symmetry = symmetry
        # self.features = features
        self.content_dict = content_dict
        self.p_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(self.embed_d * 2, 1), requires_grad=True)
        self.GCN_W1_user = torch.nn.Parameter(torch.FloatTensor(self.GCN_in, self.GCN_out1), requires_grad=True)
        self.GCN_W2_user = torch.nn.Parameter(torch.FloatTensor(self.GCN_out1, self.GCN_out2), requires_grad=True)
        self.GCN_W1_post = torch.nn.Parameter(torch.FloatTensor(self.GCN_in, self.GCN_out1), requires_grad=True)
        self.GCN_W2_post = torch.nn.Parameter(torch.FloatTensor(self.GCN_out1, self.GCN_out2), requires_grad=True)
        self.layernorm1 = nn.LayerNorm([1,out_embed_d])
        self.layernorm2 = nn.LayerNorm([1,out_embed_d])
        self.layernorm3 = nn.LayerNorm([1,out_embed_d])
        self.layernorm4 = nn.LayerNorm([1,out_embed_d])
        # Define the initial linear hidden layer
        self.init_linear_text = nn.Linear(self.input_dim[0], self.ini_hidden_dim[0])
        self.init_linear_image = nn.Linear(self.input_dim[1], self.ini_hidden_dim[1])
        if self.use_bias:
            self.user_bias1 = nn.Parameter(torch.Tensor(self.GCN_out1), requires_grad=True)
            self.post_bias1 = nn.Parameter(torch.Tensor(self.GCN_out1), requires_grad=True)
            self.user_bias2 = nn.Parameter(torch.Tensor(self.GCN_out2), requires_grad=True)
            self.post_bias2 = nn.Parameter(torch.Tensor(self.GCN_out2), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.init_linear_other = nn.Linear(self.input_dim[2], self.ini_hidden_dim[2])
        # Define the LSTM layer
        self.LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
                                                batch_first=True,
                                                bidirectional=True, dropout=0.5)
        self.LSTM_image = eval('nn.' + rnn_type)(self.ini_hidden_dim[1], self.hidden_dim, self.num_layers,
                                                 batch_first=True,
                                                 bidirectional=True, dropout=0.5)
        self.LSTM_other = eval('nn.' + rnn_type)(self.ini_hidden_dim[2], self.hidden_dim, self.num_layers,
                                                 batch_first=True,
                                                 bidirectional=True, dropout=0.5)
        # Define same_type_agg
        self.u_init_linear = nn.Linear(self.u_input_dim, self.u_ini_hidden_dim)
        # self.u_init_dropout = nn.Dropout(p=0.2)
        self.u_LSTM = eval('nn.' + self.u_rnn_type)(self.u_ini_hidden_dim, self.u_hidden_dim, self.u_num_layers,
                                                    batch_first=True, bidirectional=True, dropout=0.5)
        self.u_linear = nn.Linear(self.u_hidden_dim * 2, self.u_output_dim)
        self.u_dropout = nn.Dropout(p=0.5)
        self.p_init_linear = nn.Linear(self.p_input_dim, self.p_ini_hidden_dim)
        # self.p_init_dropout = nn.Dropout(p=0.2)
        self.p_LSTM = eval('nn.' + self.p_rnn_type)(self.p_ini_hidden_dim, self.p_hidden_dim, self.p_num_layers,
                                                    batch_first=True, bidirectional=True, dropout=0.5)
        self.p_linear = nn.Linear(self.p_hidden_dim * 2, self.p_output_dim)
        self.p_dropout = nn.Dropout(p=0.5)
        #self.act = nn.LeakyReLU()
        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.out_dropout = nn.Dropout(p=0.35)
        self.out_linear = nn.Linear(self.out_embed_d*2, self.outemb_d)
        self.batch_norm_text = nn.BatchNorm1d(self.ini_hidden_dim[0])
        self.batch_norm_image = nn.BatchNorm1d(self.ini_hidden_dim[1])
        self.batch_norm_other = nn.BatchNorm1d(self.ini_hidden_dim[2])
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
        init.kaiming_uniform_(self.GCN_W1_user)
        init.kaiming_uniform_(self.GCN_W1_post)
        init.kaiming_uniform_(self.GCN_W2_user)
        init.kaiming_uniform_(self.GCN_W2_post)
        if self.use_bias:
            init.zeros_(self.user_bias1)
            init.zeros_(self.post_bias1)
            init.zeros_(self.user_bias2)
            init.zeros_(self.post_bias2)

    def init_adj_degree(self, neighbor_id):
        adj_dim = len(neighbor_id)
        adj = np.ones((adj_dim, adj_dim))
        degree = np.diag(np.sum(adj, axis=0))
        return adj, degree
    """
    def init_matrices(self, user_neighbor_id, post_neighbor_id, user_emb_dict, post_emb_dict):
        user_H = []
        post_H = []
        for i in user_neighbor_id:
            user_H.append(user_emb_dict[i][1])
        for j in post_neighbor_id:
            post_H.append(post_emb_dict[j][0])
        user_A, user_D = self.init_adj_degree(user_neighbor_id)
        post_A, post_D = self.init_adj_degree(post_neighbor_id)
        user_H = torch.Tensor(user_H)
        user_A = torch.Tensor(user_A)
        user_D = torch.Tensor(user_D)
        post_H = torch.Tensor(post_H)
        post_A = torch.Tensor(post_A)
        post_D = torch.Tensor(post_D)
        return user_A, user_D, user_H, post_A, post_D, post_H
    """
    
    def init_matrices(self, u_aft_rnn_dict, p_aft_rnn_dict):
        user_H = []
        post_H = []
        user_H_d = u_aft_rnn_dict.values()
        post_H_d = p_aft_rnn_dict.values()
        for i in user_H_d:
            #print(i.shape)
            i = i.view(1, i.shape[0])
            user_H.append(i)
        for j in post_H_d:
            j = j.view(1, j.shape[0])
            post_H.append(j)
        user_A, user_D = self.init_adj_degree(u_aft_rnn_dict)
        post_A, post_D = self.init_adj_degree(p_aft_rnn_dict)
        user_H = torch.cat(user_H, dim=0)
        post_H = torch.cat(post_H, dim=0)
        #print(user_H.shape)
        #user_H = torch.Tensor(user_H)
        user_A = torch.Tensor(user_A)
        user_D = torch.Tensor(user_D)
        #post_H = torch.Tensor(post_H)
        post_A = torch.Tensor(post_A)
        post_D = torch.Tensor(post_D)
        return user_A, user_D, user_H, post_A, post_D, post_H
    
    def GCN_layer(self, user_A, user_D, user_H, post_A, post_D, post_H, layer_index):
        if self.symmetry == False:
            user_in = torch.mm(torch.inverse(user_D), user_A)
            post_in = torch.mm(torch.inverse(post_D), post_A)
        else:
            user_D_head = fractional_matrix_power(user_D, -0.5)
            post_D_head = fractional_matrix_power(post_D, -0.5)
            user_D_head = torch.Tensor(user_D_head)
            post_D_head = torch.Tensor(post_D_head)
            user_in = torch.mm(torch.mm(user_D_head, user_A), user_D_head)
            post_in = torch.mm(torch.mm(post_D_head, post_A), post_D_head)
        user_in_features = torch.mm(user_in, user_H)
        post_in_features = torch.mm(post_in, post_H)
        if layer_index == 1:
            user_output = torch.mm(user_in_features, self.GCN_W1_user)
            post_output = torch.mm(post_in_features, self.GCN_W1_post)
            if self.use_bias:
                user_output += self.user_bias1
                post_output += self.post_bias1
        else:
            user_output = torch.mm(user_in_features, self.GCN_W2_user)
            post_output = torch.mm(post_in_features, self.GCN_W2_post)
            if self.use_bias:
                user_output += self.user_bias2
                post_output += self.post_bias2
        return user_output, post_output
    """
    def GCN_net(self, u_aft_rnn_dict, p_aft_rnn_dict):
        user_A, user_D, user_H, post_A, post_D, post_H = self.init_matrices(u_aft_rnn_dict, p_aft_rnn_dict)
        gcn1_user, gcn1_post = self.GCN_layer(user_A, user_D, user_H, post_A, post_D, post_H, 1)
        gcn1_user = self.relu(gcn1_user)
        gcn1_post = self.relu(gcn1_post)
        gcn2_user, gcn2_post = self.GCN_layer(user_A, user_D, gcn1_user, post_A, post_D, gcn1_post, 2)
        gcn2_user = torch.mean(gcn2_user, 0)
        gcn2_post = torch.mean(gcn2_post, 0)
        gcn2_user = gcn2_user.view(1, gcn2_user.shape[0])
        gcn2_post = gcn2_post.view(1, gcn2_post.shape[0])
        gcn2_user = self.softmax(gcn2_user)
        gcn2_post = self.softmax(gcn2_post)
        #print(gcn2_user.shape)
        #print(gcn2_post.shape)
        return gcn2_user, gcn2_post
    """    
    def GCN_net(self, u_aft_rnn_dict, p_aft_rnn_dict):
        user_A, user_D, user_H, post_A, post_D, post_H = self.init_matrices(u_aft_rnn_dict, p_aft_rnn_dict)
        gcn1_user, gcn1_post = self.GCN_layer(user_A, user_D, user_H, post_A, post_D, post_H, 1)
        gcn1_user = self.relu(gcn1_user)
        gcn1_post = self.relu(gcn1_post)
        gcn2_user, gcn2_post = self.GCN_layer(user_A, user_D, gcn1_user, post_A, post_D, gcn1_post, 2)
        gcn2_user = torch.mean(gcn2_user, 0)
        gcn2_post = torch.mean(gcn2_post, 0)
        gcn2_user = gcn2_user.view(1, gcn2_user.shape[0])
        gcn2_post = gcn2_post.view(1, gcn2_post.shape[0])
        gcn2_user = self.softmax(gcn2_user)
        gcn2_post = self.softmax(gcn2_post)
        #print(gcn2_user.shape)
        #print(gcn2_post.shape)
        return gcn2_user, gcn2_post

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
            #linear_input_text = self.batch_norm_text(linear_input_text)
            linear_input_image = self.init_linear_image(input_b)
            #linear_input_image = self.batch_norm_image(linear_input_image)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_image = linear_input_image.view(linear_input_image.shape[0], 1, linear_input_image.shape[1])
            LSTM_out_text, self.hidden_text = self.LSTM_text(linear_input_text)
            LSTM_out_text = self.layernorm1(LSTM_out_text)
            LSTM_out_image, self.hidden_image = self.LSTM_image(linear_input_image)
            LSTM_out_image = self.layernorm2(LSTM_out_image)
            concate = torch.cat((LSTM_out_text, LSTM_out_image), 1)
        if node_type == "user":
            for i in neighbor_id:
                if ("user", i) not in self.content_dict:
                    input_a.append(user_emb_dict[i][0])
                    input_b.append(user_emb_dict[i][1])
                    new_id.append(i)
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_b)
            #linear_input_text = self.batch_norm_text(linear_input_text)
            linear_input_other = self.init_linear_other(input_a)
            #linear_input_other = self.batch_norm_other(linear_input_other)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            LSTM_out_text, self.hidden_text = self.LSTM_text(linear_input_text)
            LSTM_out_text = self.layernorm3(LSTM_out_text)
            LSTM_out_other, self.hidden_other = self.LSTM_other(linear_input_other)
            LSTM_out_other = self.layernorm4(LSTM_out_other)
            concate = torch.cat((LSTM_out_text, LSTM_out_other), 1)

        # mean pooling all the states
        mean_pooling = torch.mean(concate, 1)

        return mean_pooling

    # features: list of [(id)]
    def SameType_Agg_Bi_RNN(self, neighbor_id, node_type):
        content_embedings = self.Bi_RNN(neighbor_id, node_type, post_emb_dict, user_emb_dict)
        aft_rnn_dict = {}
        if node_type == 'post':
            for i in range(len(neighbor_id)):
                aft_rnn_dict[neighbor_id[i]] = content_embedings[i]
            linear_input = self.p_init_linear(content_embedings)
            # linear_input = self.p_init_dropout(linear_input)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            LSTM_out, hidden = self.p_LSTM(linear_input)
            last_state = self.p_linear(LSTM_out)
            last_state = self.p_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)

        else:
            for i in range(len(neighbor_id)):
                aft_rnn_dict[neighbor_id[i]] = content_embedings[i]
            linear_input = self.u_init_linear(content_embedings)
            # linear_input = self.u_init_dropout(linear_input)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            LSTM_out, hidden = self.u_LSTM(linear_input)
            last_state = self.u_linear(LSTM_out)
            last_state = self.u_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
        return mean_pooling, aft_rnn_dict

    def node_het_agg(self, het_node):  # heterogeneous neighbor aggregation

        # attention module
        c_agg_batch = self.Bi_RNN([het_node.node_id], het_node.node_type, post_emb_dict, user_emb_dict)
        #print(c_agg_batch.shape)
        u_agg_batch, u_aft_rnn_dict = self.SameType_Agg_Bi_RNN(het_node.neighbors_user, "user")
        p_agg_batch, p_aft_rnn_dict = self.SameType_Agg_Bi_RNN(het_node.neighbors_post, "post")
        
        gcn2_user, gcn2_post = self.GCN_net(u_aft_rnn_dict, p_aft_rnn_dict)
        #print(gcn2_user.shape, gcn2_post.shape)
        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        u_agg_batch_2 = torch.cat((c_agg_batch, u_agg_batch), 1).view(len(c_agg_batch), c_agg_batch.shape[1]+u_agg_batch.shape[1]) #400
        p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), c_agg_batch.shape[1]+p_agg_batch.shape[1])
        u_agg_batch_2 = torch.cat((gcn2_user, u_agg_batch_2), 0)
        u_agg_batch_2 = torch.mean(u_agg_batch_2, 0, keepdims = True)

        p_agg_batch_2 = torch.cat((gcn2_post, p_agg_batch_2), 0)
        p_agg_batch_2 = torch.mean(p_agg_batch_2, 0, keepdims = True)
        concate_embed = torch.cat((c_agg_batch_2, u_agg_batch_2, p_agg_batch_2), 1).view(len(c_agg_batch), 3,
                                                                                         self.embed_d * 2)
        if het_node.node_type == "user":
            atten_w = self.act(torch.bmm(concate_embed, self.u_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.u_neigh_att.size())))
        else:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),
                                                                                             *self.p_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 3)
        # weighted combination
        #concate_embed = torch.cat((c_agg_batch, u_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), 3, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d*2)
        return weight_agg_batch

    def forward(self, x):
        #gcn2_user, gcn2_post = self.GCN_net(x.neighbors_user, x.neighbors_post, user_emb_dict, post_emb_dict)
        x = self.node_het_agg(het_node=x)
        c_embed = x.view(1, self.out_embed_d*2)
        #c_embed = torch.cat((c_embed, gcn2_user, gcn2_post), 1)
        c_embed = self.out_dropout(c_embed)
        c_embed_out = self.out_linear(c_embed)
        #c_embed_out = self.batch_norm(c_embed_out)
        predictions = self.output_act(c_embed_out)
        return predictions


def BCELoss(predictions, true_label):
    loss = nn.BCELoss()
    predictions = predictions.view(1)
    tensor_label = torch.FloatTensor(np.array([true_label]))
    loss_sum = loss(predictions, tensor_label)
    return loss_sum


def save_checkpoint(model, optimizer, save_path, epoch, val_acc):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    return model, optimizer, epoch, val_acc


def train_test(data_real, data_fake, test_size):
    y_real = range(len(data_real))
    y_fake = range(len(data_fake))
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(data_real, y_real, test_size=test_size, random_state=42)
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(data_fake, y_fake, test_size=test_size, random_state=42)
    np.savetxt('FYP_models/gcn_par2vec_modi/train_index_real.txt', y_train_real)
    np.savetxt('FYP_models/gcn_par2vec_modi/test_index_real.txt', y_test_real)
    np.savetxt('FYP_models/gcn_par2vec_modi/train_index_fake.txt', y_train_fake)
    np.savetxt('FYP_models/gcn_par2vec_modi/test_index_fake.txt', y_test_fake)
    return X_train_real, X_test_real, X_train_fake, X_test_fake


def load_train_test(data_real, data_fake, test_index_path_real='FYP_models/gcn_par2vec_modi/test_index_real.txt', test_index_path_fake = 'FYP_models/gcn_par2vec_modi/test_index_fake.txt'):
    a = np.loadtxt(test_index_path_real)
    a = a.astype('int32')
    b = np.loadtxt(test_index_path_fake)
    b = b.astype('int32')
    test_set_real = []
    train_set_real = []
    for i in range(len(data_real)):
        if i in a:
            test_set_real.append(data_real[i])
        else:
            train_set_real.append(data_real[i])
    test_set_fake = []
    train_set_fake = []
    for j in range(len(data_fake)):
        if j in b:
            test_set_fake.append(data_fake[j])
        else:
            train_set_fake.append(data_fake[j])
    return train_set_real, test_set_real, train_set_fake, test_set_fake

# split test set first
#number of fake nodes:  2309
#number of real nodes:  2346
if path.exists('FYP_models/gcn_par2vec_modi/test_index_real.txt'):
    X_train_real, X_test_real, X_train_fake, X_test_fake = load_train_test(post_nodes_real, post_nodes_fake)
else:
    X_train_real, X_test_real, X_train_fake, X_test_fake = train_test(post_nodes_real, post_nodes_fake, 0.1)

# Shuffle the order in post nodes
train_val = X_train_real + X_train_fake
test_set = X_test_real + X_test_fake
np.random.shuffle(train_val)
np.random.shuffle(test_set)

# K-fold validation index
train_index = []
val_index = []
num_splits = 9
kfold = KFold(num_splits, True, 1)
for train, val in kfold.split(train_val):
    train_index.append(train)
    val_index.append(val)

# Initialize parameters
lr = 0.005
num_epoch = 40
num_folds = 1
batch_size = 4
PATH = 'FYP_models/gcn_par2vec_modi/'

print('Start training')

for fold in range(num_folds):
    print("Start for fold", fold + 1)
    best_val = 0
    running_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0
    best_epoch = 0
    net = Het_GNN(input_dim=[768, 512, 145], ini_hidden_dim=[180, 180, 180], u_hidden_dim=250, u_ini_hidden_dim=350,
                  p_hidden_dim=300, p_ini_hidden_dim=450,
                  hidden_dim=100, batch_size=1, u_input_dim=200,
                  u_batch_size=1, u_output_dim=100, u_num_layers=1, u_rnn_type='LSTM', p_input_dim=200,
                  p_batch_size=1, p_output_dim=100, p_num_layers=1,
                  p_rnn_type='LSTM', out_embed_d=110, outemb_d=1, GCN_out1= 200)

    net.init_weights()
    print(net)
    # Set up optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.05)

    for epoch in range(num_epoch):
        print('Epoch:', epoch + 1)
        c = 0.0
        running_loss = 0.0
        v = 0.0
        val_loss = 0.0
        real_count = 0.0
        fake_count = 0.0
        real_true = 0.0
        fake_true = 0.0
        # generate train and test set for current epoch
        train_set = []
        val_set = []
        for t_index in train_index[fold]:
            train_set.append(train_val[t_index])
        for v_index in val_index[fold]:
            val_set.append(train_val[v_index])
        net.train()
        for i in range(len(train_set)):
            optimizer.zero_grad()
            output = net(train_set[i])
            if (output.item() >= 0.5 and train_set[i].label == 1) or (output.item() < 0.5 and train_set[i].label == 0):
                c += 1
            cur_loss = BCELoss(predictions=output, true_label=train_set[i].label)
            running_loss += cur_loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('Fold: %d, Epoch: %d, step: %5d, loss: %.4f, acc: %.4f' %
                      (fold + 1, epoch + 1, i + 1, running_loss / 100, c / 100))
                running_loss = 0.0
                c = 0.0
            if i % batch_size == 0:
                loss = Variable(torch.zeros(1), requires_grad=True)
            # loss = loss + cur_loss
            x = random.random()
            if (output.item() < 0.5 and train_set[i].label == 1):
                loss = loss + (1+x)*cur_loss
            else:
                loss = loss + cur_loss
            if i % batch_size == (batch_size - 1):
                loss = loss / batch_size
                loss.backward()
                optimizer.step()
        net.eval()
        for j in range(len(val_set)):
            output = net(val_set[j])
            if val_set[j].label == 1:
                real_count += 1
                if output.item() >= 0.5:
                    real_true += 1
            else:
                fake_count += 1
                if output.item() < 0.5:
                    fake_true += 1
            if (output.item() >= 0.5 and val_set[j].label == 1) or (output.item() < 0.5 and val_set[j].label == 0):
                v += 1
            vloss = BCELoss(predictions=output, true_label=val_set[j].label)
            x = random.random()
            if (output.item() < 0.5 and val_set[j].label == 1):
                val_loss += (1+x)*vloss.item()
            else:
                val_loss += vloss.item()
        val_acc = v / len(val_set)
        real_precision = real_true / (real_true + fake_count - fake_true)
        fake_precision = fake_true / (fake_true + real_count - real_true)
        real_recall = real_true / real_count
        fake_recall = fake_true / fake_count
        real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)
        fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall)
        print('Validation loss: %.4f, Validation accuracy: %.4f' % (val_loss / len(val_set), val_acc))
        print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f' % (real_precision, real_recall, real_f1))
        print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f' % (fake_precision, fake_recall, fake_f1))
        """
        if val_loss / len(val_set) < best_val:
            print('Update model at epoch:', epoch + 1)
            cur_PATH = PATH + 'best_model' + '_' + str(fold) + '.tar'
            save_checkpoint(net, optimizer, cur_PATH, epoch + 1, val_acc)
            best_val = val_loss / len(val_set)
            best_epoch = epoch
        """
        if val_acc > best_val:
            print('Update model at epoch:', epoch + 1)
            cur_PATH = PATH + 'best_model_test' + '_' + str(fold) + '.tar'
            save_checkpoint(net, optimizer, cur_PATH, epoch + 1, val_acc)
            best_val = val_acc
            best_epoch = epoch
        scheduler.step()
        if epoch - best_epoch >= 3:
            break
    print('Finish training')

print('==============================================================')


# In[ ]:


# Init net and optimizer skeletons
best_models = []
num_folds = 1
net = Het_GNN(input_dim=[768, 512, 145], ini_hidden_dim=[180, 180, 180], u_hidden_dim=250, u_ini_hidden_dim=350,
                  p_hidden_dim=300, p_ini_hidden_dim=450,
                  hidden_dim=100, batch_size=1, u_input_dim=200,
                  u_batch_size=1, u_output_dim=100, u_num_layers=1, u_rnn_type='LSTM', p_input_dim=200,
                  p_batch_size=1, p_output_dim=100, p_num_layers=1,
                  p_rnn_type='LSTM', out_embed_d=110, outemb_d=1, GCN_out1= 200)
net.init_weights()
optimizer = optim.SGD(net.parameters(), lr=lr)

for count in range(num_folds):
    cur_PATH = PATH + 'best_model_test_95.5%' + '_' + str(count) + '.tar'
    net, optimizer, epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)
    print(best_val)
    net.eval()
    best_models.append(net)

t = 0.0
test_loss = 0.0
real_count = 0.0
fake_count = 0.0
real_true = 0.0
fake_true = 0.0
for k in range(len(test_set)):
    output = 0.0
    avg_tloss = 0.0
    for fold in range(num_folds):
        result = best_models[fold](test_set[k])
        output += result.item()
        tloss = BCELoss(predictions=result, true_label=test_set[k].label)
        avg_tloss += tloss.item()
    output /= num_folds
    avg_tloss /= num_folds
    test_loss += avg_tloss
    if (output >= 0.5 and test_set[k].label == 1) or (output < 0.5 and test_set[k].label == 0):
        t += 1
    if test_set[k].label == 1:
        real_count += 1
        if output >= 0.5:
            real_true += 1
    else:
        fake_count += 1
        if output < 0.5:
            fake_true += 1

real_precision = real_true / (real_true + fake_count - fake_true)
fake_precision = fake_true / (fake_true + real_count - real_true)
real_recall = real_true / real_count
fake_recall = fake_true / fake_count
real_f1 = 2 * real_precision * real_recall / (real_precision + real_recall)
fake_f1 = 2 * fake_precision * fake_recall / (fake_precision + fake_recall)
print('Test loss: %.4f, Test accuracy: %.4f' % (test_loss / len(test_set), t / len(test_set)))
print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f' % (real_precision, real_recall, real_f1))
print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f' % (fake_precision, fake_recall, fake_f1))


# In[ ]:




