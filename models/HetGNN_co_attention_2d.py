#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.optim as optim
from os import path
from torch.autograd import Variable


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_post=[], neighbor_list_user=[], label=None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label  # only post node, user node = default = None
        self.neighbors_user = neighbor_list_user  # [(id)]
        self.neighbors_post = neighbor_list_post


def data_loader(pathway='F:/post_nodes/', node_type="post"):
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
                if j % 189 == 0:
                    id_ = Lines[j].split()
                    post_id.append(int(float(id_[0])))
                    normalized_text = []
                    normalized_image = []
                    embed = []
                if j % 189 == 1:
                    #print(Lines[j].split())
                    label_ = Lines[j].split()
                    post_label.append(int(float(label_[0])))
                if j % 189 >= 2 and j % 189 <= 137:
                    normalized_text.append(list(map(float, Lines[j].split())))
                    if j % 189 == 137:
                        text_matrix = np.stack(normalized_text)
                        embed.append(text_matrix)
                if j % 189 >= 138 and j % 189 <= 186:
                    normalized_image.append(list(map(float, Lines[j].split())))
                    if j % 189 == 186:
                        image_matrix = np.stack(normalized_image)
                        embed.append(image_matrix)
                if j % 189 == 186:
                    post_embed.append(embed)
                if j % 189 == 187:
                    post_p_neigh.append(list(map(int, map(float, Lines[j].split()))))
                if j % 189 == 188:
                    post_u_neigh.append(list(map(int, map(float, Lines[j].split()))))
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


post_nodes = data_loader(pathway='weibo_post_2d_m/', node_type="post")
user_nodes = data_loader(pathway='FYP_data/normalized_user_nodes/', node_type="user")

key = 12695585126
for node in post_nodes:
    if key in node.neighbors_post:
        print("bad")
        node.neighbors_post.remove(key)
        



post_emb_dict = {}
user_emb_dict = {}
for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb


# In[3]:


print(len(post_emb_dict))
print(len(post_nodes))


# In[4]:


class Het_GNN(nn.Module):
    # features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim, batch_size, concate_d, out_linear_d, outemb_d, content_dict={}, num_layers=1,
                 rnn_type='LSTM', embed_d=200, dim_d_t = 300, dim_d_i = 512, dim_k_ti = 1, dim_d_tp = 3000, dim_k_tp = 1, dim_d_tu = 30, dim_k_tu = 1):
        super(Het_GNN, self).__init__()
        self.input_dim = input_dim
        self.ini_hidden_dim = ini_hidden_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_d = embed_d
        self.out_embed_d = concate_d
        self.out_linear_d = out_linear_d
        self.outemb_d = outemb_d
        self.dim_d_t = dim_d_t
        self.dim_d_i = dim_d_i
        self.dim_k_ti = dim_k_ti
        self.dim_d_u = 4*self.hidden_dim
        self.dim_k_tu = dim_k_tu
        self.dim_d_p = 370*self.hidden_dim
        self.dim_k_tp = dim_k_tp
        # self.features = features
        self.content_dict = content_dict
        # Define the initial linear hidden layer
        self.init_linear_text = nn.Linear(self.input_dim[0], self.ini_hidden_dim[0])
        self.init_linear_image = nn.Linear(self.input_dim[1], self.ini_hidden_dim[1])
        self.init_linear_other = nn.Linear(self.input_dim[2], self.ini_hidden_dim[2])
        # Define the LSTM layer
        self.LSTM_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers,
                                                batch_first=True,
                                                bidirectional=True, dropout=0.1)
        self.LSTM_image = eval('nn.' + rnn_type)(self.ini_hidden_dim[1], self.hidden_dim, self.num_layers,
                                                 batch_first=True,
                                                 bidirectional=True, dropout=0.1)
        self.LSTM_other = eval('nn.' + rnn_type)(self.ini_hidden_dim[2], self.hidden_dim, self.num_layers,
                                                 batch_first=True,
                                                 bidirectional=True, dropout=0.1)
        self.Wti = torch.nn.Parameter(torch.FloatTensor(self.dim_d_t, self.dim_d_i), requires_grad=True)
        self.Wt1 = torch.nn.Parameter(torch.FloatTensor(self.dim_k_ti, self.dim_d_t), requires_grad=True)
        self.Wi = torch.nn.Parameter(torch.FloatTensor(self.dim_k_ti, self.dim_d_i), requires_grad=True)
        self.wht1 = torch.nn.Parameter(torch.FloatTensor(1, self.dim_k_ti), requires_grad=True)
        self.whi = torch.nn.Parameter(torch.FloatTensor(1, self.dim_k_ti), requires_grad=True)
        nn.init.xavier_normal_(self.Wti)
        nn.init.xavier_normal_(self.Wt1)
        nn.init.xavier_normal_(self.Wi)
        nn.init.xavier_normal_(self.wht1)
        nn.init.xavier_normal_(self.whi)

        self.Wtu = torch.nn.Parameter(torch.FloatTensor(self.dim_d_t, self.dim_d_u), requires_grad=True)
        self.Wt2 = torch.nn.Parameter(torch.FloatTensor(self.dim_k_tu, self.dim_d_t), requires_grad=True)
        self.Wu = torch.nn.Parameter(torch.FloatTensor(self.dim_k_tu, self.dim_d_u), requires_grad=True)
        self.wht2 = torch.nn.Parameter(torch.FloatTensor(1, self.dim_k_tu), requires_grad=True)
        self.whu = torch.nn.Parameter(torch.FloatTensor(1, self.dim_k_tu), requires_grad=True)
        nn.init.xavier_normal_(self.Wtu)
        nn.init.xavier_normal_(self.Wt2)
        nn.init.xavier_normal_(self.Wu)
        nn.init.xavier_normal_(self.wht2)
        nn.init.xavier_normal_(self.whu)

        self.Wtp = torch.nn.Parameter(torch.FloatTensor(self.dim_d_t, self.dim_d_p), requires_grad=True)
        self.Wt3 = torch.nn.Parameter(torch.FloatTensor(self.dim_k_tp, self.dim_d_t), requires_grad=True)
        self.Wp = torch.nn.Parameter(torch.FloatTensor(self.dim_k_tp, self.dim_d_p), requires_grad=True)
        self.wht3 = torch.nn.Parameter(torch.FloatTensor(1, self.dim_k_tp), requires_grad=True)
        self.whp = torch.nn.Parameter(torch.FloatTensor(1, self.dim_k_tp), requires_grad=True)
        nn.init.xavier_normal_(self.Wtp)
        nn.init.xavier_normal_(self.Wt3)
        nn.init.xavier_normal_(self.Wp)
        nn.init.xavier_normal_(self.wht3)
        nn.init.xavier_normal_(self.whp)
        """
        self.feature_trans_img = nn.Linear(512, self.dim_d_ti)
        self.feature_trans_text_i = nn.Linear(300, self.dim_d_ti)
        self.feature_trans_text_p = nn.Linear(300, self.dim_d_tp)
        self.feature_trans_text_u = nn.Linear(300, self.dim_d_tu)
        self.feature_trans_user = nn.Linear(4*self.hidden_dim, self.dim_d_tu)
        self.feature_trans_post = nn.Linear(370*self.hidden_dim, self.dim_d_tp)
        """
        self.out_linear = nn.Linear(self.out_embed_d, self.out_linear_d)
        self.act = nn.ReLU()
        self.out_dropout = nn.Dropout(p=0.1)
        self.fully_connect = nn.Linear(self.out_linear_d, self.outemb_d)
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def coattention(self, text_feature, image_feature, Wb, Wimg, Wtext, Whimg, Whtext, dim_k):
        """
        Parameters
        ----------
        text_feature : tensors, text embeddings
        image_feature : tensors, image embeddings
        Wb : Wsg
        Wimg : Wg
        Wtext : Ws
        Whimg : whg
        Whtext : whs

        All of the weights above should be learnable parameters and need to be initialized before passing to the function

        Returns
        -------
        context_text : weighted text context vectors
        context_img : weighted image context vectors

        """

        # C here could be referenced as F in the GCAN paper :]
        #print(text_feature.shape)
        C = torch.matmul(torch.transpose(text_feature,0,1), Wb)
        #print(C.shape)
        C = torch.matmul(C, image_feature)
        #print(C.shape)
        C = torch.tanh(C)
        WtT = torch.matmul(Wtext, text_feature)
        WtT_C = torch.matmul(WtT, C)
        WiI = torch.matmul(Wimg, image_feature)
        WiI_C = torch.matmul(WiI, torch.transpose(C,0,1))
        H_text = WtT + WiI_C  # (Wv)V + ((Wq)Q)C
        H_text = torch.tanh(H_text)
        H_img = WiI + WtT_C
        H_img = torch.tanh(H_img)
        #print(image_feature.shape)
        #print(H_text.shape)
        #print(H_img.shape)
        #print(Whimg.shape)
        softmax = nn.Softmax()
        Himg_w = nn.Linear(dim_k, 1)
        Htext_w = nn.Linear(dim_k, 1)
        #a_img = softmax(H_img.T * Whimg)
        # attention probabilities of each word qt
        #a_text = softmax(H_text.T * Whtext)
        a_img = softmax(Himg_w(H_img.T))
        a_text = softmax(Htext_w(H_text.T))
        context_img = a_img.T * image_feature
        context_text = a_text.T * text_feature
        #print(context_img.shape)
        #print(context_text.shape)
        context_img = torch.sum(context_img, dim=1)   #200
        context_text = torch.sum(context_text, dim=1) #200
        #print(context_img.shape)
        #print(context_text.shape)
        return context_text, context_img

    def get_input_features(self, neighbor_id, post_emb_dict):
        input_text = []
        input_image = []
        for i in neighbor_id:
            input_text.append(post_emb_dict[i][0])
            input_image.append(post_emb_dict[i][1])
        input_text = torch.Tensor(input_text)
        input_image = torch.Tensor(input_image)
        return input_text, input_image

    def Bi_RNN(self, neighbor_id, node_type, post_emb_dict, user_emb_dict):
        # Forward pass through initial hidden layer
        input_a = []
        input_b = []
        if node_type == "post":
            for i in neighbor_id:
                input_a.append(post_emb_dict[i][0])
                input_b.append(post_emb_dict[i][1])
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_a) #[5, 136, 150]
            linear_input_image = self.init_linear_image(input_b) #[5, 49, 150]
            #print(linear_input_text.shape) 
            #print(linear_input_image.shape) 
            linear_input_text = linear_input_text.view(linear_input_text.shape[1], linear_input_text.shape[0], linear_input_text.shape[2])
            linear_input_image = linear_input_image.view(linear_input_image.shape[1], linear_input_image.shape[0], linear_input_image.shape[2])
            LSTM_out_text, self.hidden_text = self.LSTM_text(linear_input_text)
            LSTM_out_image, self.hidden_image = self.LSTM_image(linear_input_image)
            #print(LSTM_out_image.shape)
            #print(LSTM_out_text.shape)
            concate = torch.cat((LSTM_out_text, LSTM_out_image), 0)
            concate = concate.view(concate.shape[1], concate.shape[0], concate.shape[2])
            #print(concate.shape)
            concate = concate.view(concate.shape[0], -1)
            #print(concate.shape)
        if node_type == "user":
            for i in neighbor_id:
                input_a.append(user_emb_dict[i][0])
                input_b.append(user_emb_dict[i][1])
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_b)
            linear_input_other = self.init_linear_other(input_a)
            #print(linear_input_text.shape)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            LSTM_out_text, self.hidden_text = self.LSTM_text(linear_input_text)
            LSTM_out_other, self.hidden_other = self.LSTM_other(linear_input_other)
            concate = torch.cat((LSTM_out_text, LSTM_out_other), 1)
            concate = concate.view(concate.shape[0], -1)

        return concate

    def node_het_agg(self, het_node):  # heterogeneous neighbor aggregation

        # attention module
        user_neighbor = self.Bi_RNN(het_node.neighbors_user, "user", post_emb_dict, user_emb_dict) #20*400
        post_neighbor = self.Bi_RNN(het_node.neighbors_post, "post", post_emb_dict, user_emb_dict) #5*400
        text_feature, image_feature = self.get_input_features([het_node.node_id], post_emb_dict) #1*300, 1*512
        """
        print(user_neighbor.shape)
        print(post_neighbor.shape)
        print(text_feature.shape)
        print(image_feature.shape)
        """
        #user_neighbor = torch.transpose(user_neighbor, 0, 1)
        #post_neighbor = torch.transpose(post_neighbor, 0, 1)
        text_feature =  text_feature.view(text_feature.shape[1], -1)
        image_feature = image_feature.view(image_feature.shape[1], -1)
        #print(text_feature.shape)
        user_neighbor = torch.transpose(user_neighbor, 0, 1)
        post_neighbor = torch.transpose(post_neighbor, 0, 1)
        text_feature = torch.transpose(text_feature, 0, 1)
        image_feature = torch.transpose(image_feature, 0, 1)
        
        """
        text_feature_ti = self.feature_trans_text_i(text_feature)
        text_feature_tp = self.feature_trans_text_p(text_feature)
        text_feature_tu = self.feature_trans_text_u(text_feature)
        image_feature = self.feature_trans_img(image_feature)
        user_neighbor = self.feature_trans_user(user_neighbor)
        post_neighbor  = self.feature_trans_post(post_neighbor)
        
        user_neighbor = torch.transpose(user_neighbor, 0, 1)
        post_neighbor = torch.transpose(post_neighbor, 0, 1)
        text_feature_ti = torch.transpose(text_feature_ti, 0, 1)
        text_feature_tu = torch.transpose(text_feature_tu, 0, 1)
        text_feature_tp = torch.transpose(text_feature_tp, 0, 1)
        image_feature = torch.transpose(image_feature, 0, 1)
        """
        #print(user_neighbor.shape)
        #print(post_neighbor.shape)
        #print(text_feature.shape)
        #print(image_feature.shape)
        
        co_text1, co_image = self.coattention(text_feature=text_feature, image_feature=image_feature,
                                         Wb=self.Wti, Wimg=self.Wi, Wtext=self.Wt1, Whimg=self.whi, Whtext=self.wht1, dim_k = self.dim_k_ti)

        co_text2, co_user = self.coattention(text_feature=text_feature, image_feature=user_neighbor,
                                        Wb=self.Wtu, Wimg=self.Wu, Wtext=self.Wt2, Whimg=self.whu, Whtext=self.wht2, dim_k = self.dim_k_tu)
        co_text3, co_post = self.coattention(text_feature=text_feature, image_feature=post_neighbor,
                                        Wb=self.Wtp, Wimg=self.Wp, Wtext=self.Wt3, Whimg=self.whp, Whtext=self.wht3, dim_k = self.dim_k_tp)
        co_text = torch.stack([co_text1, co_text2, co_text3])
        #print(co_text.shape)
        
        co_text_mean = torch.mean(co_text, 0) 
        context_vector = torch.cat([co_text_mean, co_image, co_user, co_post])
        #context_vector = torch.cat([co_text1, co_image, co_text2, co_user, co_text3, co_post])
        #print(context_vector.shape)
        return context_vector

    def output(self, c_embed_batch):

        batch_size = 1
        # make c_embed 3D tensor. Batch_size * 1 * embed_d
        c_embed = c_embed_batch.view(batch_size, 1, self.out_embed_d)
        c_embed = self.out_linear(c_embed)
        c_embed = self.act(c_embed)
        c_embed = self.out_dropout(c_embed)
        c_embed_out = self.fully_connect(c_embed)
        predictions = self.output_act(c_embed_out)  # log(1/(1+exp(-x)))    sigmoid = 1/(1+exp(-x))
        return predictions

    def forward(self, x):
        x = self.node_het_agg(het_node=x)
        x = self.output(c_embed_batch=x)
        return x


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


def train_test(data, train_size, test_size):
    y = range(train_size + test_size)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=42)
    np.savetxt('FYP_models/lstm_co_attention/train_index.txt', y_train)
    np.savetxt('FYP_models/lstm_co_attention/test_index.txt', y_test)
    return X_train, X_test


def load_train_test(data, test_index_path='FYP_data/test_index.txt'):
    a = np.loadtxt(test_index_path)
    a = a.astype('int32')
    test_set = []
    train_set = []
    for i in range(len(data)):
        if i in a:
            test_set.append(data[i])
        else:
            train_set.append(data[i])
    return train_set, test_set


# split test set first
if path.exists('FYP_models/lstm_co_attention/test_index.txt'):
    train_val, test_set = load_train_test(post_nodes)
else:
    train_val, test_set = train_test(post_nodes, 3604, 1046)



# Shuffle the order in post nodes
np.random.shuffle(train_val)

# K-fold validation index
train_index = []
val_index = []
num_folds = 8
kfold = KFold(num_folds, True, 1)
for train, val in kfold.split(train_val):
    train_index.append(train)
    val_index.append(val)

# best_models = []
# Initialize parameters
lr = 0.00005
num_epoch = 40
batch_size = 8
PATH = 'FYP_models/lstm_co_attention/'
num_itr = 1
print('Start training')

for fold in range(num_itr):
    print("Start for fold", fold)
    best_val = 0.0
    running_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0
    best_epoch = 0
    net = Het_GNN(input_dim=[300, 512, 12], ini_hidden_dim=[200, 200, 200], hidden_dim=150, batch_size=1, concate_d=56912, out_linear_d=200, outemb_d=1)
    net.init_weights()

    # Set up optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epoch):
        print('Epoch:', epoch + 1)
        c = 0.0
        running_loss = 0.0
        v = 0.0
        val_loss = 0.0
        t = 0.0
        test_loss = 0.0
        # generate train and test set for current epoch
        train_set = []
        val_set = []
        for t_index in train_index[fold]:
            train_set.append(train_val[t_index])
        for v_index in val_index[fold]:
            val_set.append(train_val[v_index])
        np.random.shuffle(train_set)
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
            loss = loss + cur_loss
            if i % batch_size == (batch_size - 1):
                loss = loss / batch_size
                loss.backward()
                optimizer.step()
        net.eval()
        for j in range(len(val_set)):
            output = net(val_set[j])
            if (output.item() >= 0.5 and val_set[j].label == 1) or (output.item() < 0.5 and val_set[j].label == 0):
                v += 1
            vloss = BCELoss(predictions=output, true_label=val_set[j].label)
            val_loss += vloss.item()
        val_acc = v / len(val_set)
        print('Validation loss: %.4f, Validation accuracy: %.4f' % (val_loss / len(val_set), val_acc))
        for j in range(len(test_set)):
            output = net(test_set[j])
            if (output.item() >= 0.5 and test_set[j].label == 1) or (output.item() < 0.5 and test_set[j].label == 0):
                t += 1
            tloss = BCELoss(predictions=output, true_label=test_set[j].label)
            test_loss += tloss.item()
        test_acc = t / len(test_set)
        print('Test loss: %.4f, Test accuracy: %.4f' % (test_loss / len(test_set), test_acc))
        if val_acc > best_val:
            print('Update model at epoch:', epoch + 1)
            cur_PATH = PATH + 'best_model' + '_' + str(fold) + '.tar'
            save_checkpoint(net, optimizer, cur_PATH, epoch + 1, val_acc)
            best_val = val_acc
            best_epoch = epoch
        scheduler.step()
        if epoch - best_epoch >= 8:
            break

    print('Finish training')

print('==============================================================')
# Init net and optimizer skeletons
# num_folds = 3
'''
num_itr = 5
best_models = []
net = Het_GNN(input_dim = [300, 512, 12], ini_hidden_dim = [150, 150, 150], hidden_dim=100, batch_size=1, u_input_dim=200, u_hidden_dim=150, u_ini_hidden_dim=150,
                                   u_batch_size=1, u_output_dim=200, u_num_layers=1, u_rnn_type='LSTM', p_input_dim=200,
                                   p_hidden_dim=150, p_ini_hidden_dim=150, p_batch_size=1, p_output_dim=200, p_num_layers=1,
                                    p_rnn_type='LSTM',out_embed_d=200, outemb_d=1)
net.init_weights()
optimizer = optim.SGD(net.parameters(), lr=lr)


for count in range(num_itr):
    cur_PATH = PATH + 'best_model'+'_'+str(count)+'.tar'
    net, optimizer, epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)
    print(best_val)
    net.eval()
    best_models.append(net)

t = 0.0
test_loss = 0.0
for k in range(len(test_set)):
    output = 0.0
    avg_tloss = 0.0
    for fold in range(num_itr):
        result = best_models[fold](test_set[k])
        output += result.item()
        tloss = BCELoss(predictions=result, true_label=test_set[k].label)
        avg_tloss += tloss.item()
    output /= num_itr
    avg_tloss /= num_itr
    test_loss += avg_tloss
    if (output >= 0.5 and test_set[k].label == 1) or (output < 0.5 and test_set[k].label == 0):
        t += 1

print('Test loss: %.4f, Test accuracy: %.4f'% (test_loss/len(test_set), t/len(test_set)))
'''


# In[ ]:





# In[ ]:





# In[ ]:




