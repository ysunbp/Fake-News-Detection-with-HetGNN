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


class Bi_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=11, num_layers=1, rnn_type='LSTM'):
        super(Bi_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the initial linear hidden layer
        self.init_linear_text = nn.Linear(self.input_dim[0], self.input_dim[0])
        self.init_linear_image = nn.Linear(self.input_dim[1], self.input_dim[1])
        self.init_linear_other = nn.Linear(self.input_dim[2], self.input_dim[2])
        # Define the LSTM layer
        self.lstm_text = eval('nn.' + rnn_type)(self.input_dim[0], self.hidden_dim, self.num_layers, batch_first=True,
                                                bidirectional=True)
        self.lstm_image = eval('nn.' + rnn_type)(self.input_dim[1], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True)
        self.lstm_other = eval('nn.' + rnn_type)(self.input_dim[2], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)

    def init_hidden(self):
        # initialise our hidden state
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through initial hidden layer
        linear_input_text = self.init_linear_text(input[0])
        linear_input_image = self.init_linear_image(input[1])
        linear_input_other = self.init_linear_other(input[2])

        lstm_out_text, self.hidden_text = self.lstm_text(linear_input_text)
        lstm_out_image, self.hidden_image = self.lstm_image(linear_input_image)
        lstm_out_other, self.hidden_other = self.lstm_other(linear_input_other)

        concate = torch.Tensor([lstm_out_text, lstm_out_image, lstm_out_other])
        # mean pooling all the states
        mean_pooling = torch.mean(concate, 0)
        return mean_pooling


class SameType_Agg_Bi_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers=1, rnn_type='LSTM'):
        super(Bi_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the initial linear hidden layer
        self.init_linear = nn.Linear(self.input_dim, self.input_dim)

        # Define the LSTM layer
        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True,
                                           bidirectional=True)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim * 2, output_dim)

    def init_hidden(self):
        # initialise our hidden state
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through initial hidden layer
        linear_input = self.init_linear(input)
        lstm_out, self.hidden = self.lstm(linear_input)
        # mean pooling all the states
        mean_pooling = torch.mean(last_state, 0)
        return mean_pooling


def node_het_agg(self, node_type): #heterogeneous neighbor aggregation
        #attention module
        if node_type == "user":
            c_agg_batch = Bi_RNN("user")
        else:
            c_agg_batch = Bi_RNN("post")
        u_agg_batch = SameType_Agg_Bi_RNN("user")
        p_agg_batch = SameType_Agg_Bi_RNN("post")

        c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        u_agg_batch_2 = torch.cat((c_agg_batch, u_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
        p_agg_batch_2 = torch.cat((c_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

        #compute weights
        concate_embed = torch.cat((c_agg_batch_2, u_agg_batch_2, p_agg_batch_2), 1).view(len(c_agg_batch), 4, self.embed_d * 2)
        if node_type == "user":
            atten_w = self.act(torch.bmm(concate_embed, self.u_neigh_att.unsqueeze(0).expand(len(c_agg_batch),*self.u_neigh_att.size())))
        else:
            atten_w = self.act(torch.bmm(concate_embed, self.p_neigh_att.unsqueeze(0).expand(len(c_agg_batch),*self.p_neigh_att.size())))
        atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)

        #weighted combination
        concate_embed = torch.cat((c_agg_batch, u_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), 4, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

        return weight_agg_batch

'''def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
        embed_d = self.embed_d
        # batch processing
        # nine cases for academic data (user, paper)
        if triple_index == 0:
            c_agg = self.node_het_agg(c_id_batch, "user")
            p_agg = self.node_het_agg(pos_id_batch, "user")
            n_agg = self.node_het_agg(neg_id_batch, "user")
        elif triple_index == 1:
            c_agg = self.node_het_agg(c_id_batch, "user")
            p_agg = self.node_het_agg(pos_id_batch, "post")
            n_agg = self.node_het_agg(neg_id_batch, "post")
        elif triple_index == 2:
            c_agg = self.node_het_agg(c_id_batch, "post")
            p_agg = self.node_het_agg(pos_id_batch, "user")
            n_agg = self.node_het_agg(neg_id_batch, "user")
        elif triple_index == 4:
            c_agg = self.node_het_agg(c_id_batch, "post")
            p_agg = self.node_het_agg(pos_id_batch, "post")
            n_agg = self.node_het_agg(neg_id_batch, "post")
        elif triple_index == 5:  # save learned node embedding
            embed_file = open(self.args.data_path + "node_embedding.txt", "w")
            save_batch_s = self.args.mini_batch_s
            for i in range(2):
                if i == 0:
                    batch_number = int(len(self.u_train_id_list) / save_batch_s)
                else:
                    batch_number = int(len(self.p_train_id_list) / save_batch_s)
                for j in range(batch_number):
                    if i == 0:
                        id_batch = self.u_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, "user")
                    else:
                        id_batch = self.p_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
                        out_temp = self.node_het_agg(id_batch, "post")
                    out_temp = out_temp.data.cpu().numpy()
                    for k in range(len(id_batch)):
                        index = id_batch[k]
                        if i == 0:
                            embed_file.write('u' + str(index) + " ")
                        else:
                            embed_file.write('p' + str(index) + " ")
                        for l in range(embed_d - 1):
                            embed_file.write(str(out_temp[k][l]) + " ")
                        embed_file.write(str(out_temp[k][-1]) + "\n")

                if i == 0:
                    id_batch = self.u_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, "user")
                else:
                    id_batch = self.p_train_id_list[batch_number * save_batch_s: -1]
                    out_temp = self.node_het_agg(id_batch, "post")
                out_temp = out_temp.data.cpu().numpy()
                for k in range(len(id_batch)):
                    index = id_batch[k]
                    if i == 0:
                        embed_file.write('u' + str(index) + " ")
                    else:
                        embed_file.write('p' + str(index) + " ")
                    for l in range(embed_d - 1):
                        embed_file.write(str(out_temp[k][l]) + " ")
                    embed_file.write(str(out_temp[k][-1]) + "\n")
            embed_file.close()
            return [], [], []

        return c_agg, p_agg, n_agg'''
