#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool  # need to install torch_geometric
#from torch.nn import MultiheadAttention
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.optim as optim
from os import path
from torch.autograd import Variable

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# In[2]:


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
    np.savetxt('F:\\FYP_data\\train_index.txt', y_train)
    np.savetxt('F:\\FYP_data\\test_index.txt', y_test)
    return X_train, X_test


def load_train_test(data, test_index_path='/Users/jessica/Desktop/experiment_results/test_index.txt'):
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


# In[3]:


class SemanticAttention(nn.Module):  # the semantic attention (weight of different type neighbors)
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False) # i modified out_feature to fit the out_act layer
        )

    def forward(self, z):
        w = self.project(z)
        #print("the shape of w ", w.shape)
        beta = torch.softmax(w, dim=1)
        #print("the shape of beta[0] is ", beta[0])
        #print("the shape of z[0] is ", z[0].shape)
        #print("the shape of beta*z", (beta*z).sum(1).shape)
        #print("if i only take the first one ", torch.matmul(beta[0].reshape(1,4), z[0]).shape)
        return (beta * z).sum(1)


# In[4]:


class AttentionLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(AttentionLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        # GAT layer corresponds to type attention layer in HetGAN
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_channels=in_size, out_channels=out_size, heads=layer_num_heads,
                                           dropout=dropout))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, h, gs):
        semantic_embeddings = []
        # print(gs)
        for i, g in enumerate(gs):  # (type, neighbor_node)
            #print(g.shape)
            #print(h.shape)
            #print(self.gat_layers[i](h, g))
            semantic_embeddings.append(self.gat_layers[i](h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        # print("the shape of semantic embedding ", semantic_embeddings.shape)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# In[5]:



class AttentionNet(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads=4, num_layer=5, dropout=0.2):
        super(AttentionNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(AttentionLayer(num_meta_paths, in_size, hidden_size, num_heads, dropout))
        for l in range(1, num_layer - 1):
            self.layers.append(AttentionLayer(num_meta_paths, hidden_size * num_heads,
                                              hidden_size, num_heads, dropout))
        self.layers.append(
            AttentionLayer(num_meta_paths, hidden_size * num_heads, out_size, num_heads, dropout))

    def forward(self, g, h):
        #print("problem after attention net")
        for gnn in self.layers:
            h = gnn(h, g)

        return h  # the final hidden state


# In[6]:


class Het_Node():
    def __init__(self, node_type, node_id, embed, neighbor_list_post=[], neighbor_list_user=[], label=None):
        self.node_type = node_type
        self.node_id = node_id
        self.emb = embed
        self.label = label  # only post node, user node = default = None
        self.neighbors_user = neighbor_list_user  # [(id)]
        self.neighbors_post = neighbor_list_post

        if node_type == "user":
            self.index_self_to_post = None
            self.index_post_to_self = None
            self.index_self_to_user = None
            self.index_user_to_self = None
        elif node_type == "post":
            source_self_to_post = [0] * len(neighbor_list_post)
            target_self_to_post = list(np.arange(1, len(neighbor_list_post) + 1))

            source_post_to_self = target_self_to_post
            target_post_to_self = source_self_to_post

            source_self_to_user = [0] * len(neighbor_list_user)
            target_self_to_user = list(
                np.arange(len(neighbor_list_post) + 1, len(neighbor_list_post) + len(neighbor_list_user) + 1))

            source_user_to_self = target_self_to_user
            target_user_to_self = source_self_to_user

            self.index_self_to_post = torch.LongTensor([source_self_to_post, target_self_to_post])
            self.index_post_to_self = torch.LongTensor([source_post_to_self, target_post_to_self])
            self.index_self_to_user = torch.LongTensor([source_self_to_user, target_self_to_user])
            self.index_user_to_self = torch.LongTensor([source_user_to_self, target_user_to_self])


# In[40]:


class Het_GNN(nn.Module):
    # features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim, batch_size,
                 u_input_dim, u_hidden_dim, u_ini_hidden_dim, u_output_dim, u_num_layers,
                 p_input_dim, p_hidden_dim, p_ini_hidden_dim, p_output_dim, p_num_layers,
                 out_embed_d, outemb_d,
                 u_batch_size=1, p_batch_size=1, content_dict={}, num_layers=1, u_rnn_type='LSTM', p_rnn_type='LSTM',
                 rnn_type='LSTM', embed_d=200, num_meta_path=4, num_head=2, att_num_layers=5, att_dropout=0.3, Attlayer=1):
        super(Het_GNN, self).__init__()
        self.input_dim = input_dim
        self.ini_hidden_dim = ini_hidden_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_d = embed_d
        self.u_input_dim = u_input_dim
        self.u_hidden_dim = u_hidden_dim
        self.u_ini_hidden_dim = u_ini_hidden_dim
        self.u_batch_size = u_batch_size
        self.u_output_dim = u_output_dim
        self.u_num_layers = u_num_layers
        self.u_rnn_type = u_rnn_type
        self.p_input_dim = p_input_dim
        self.p_hidden_dim = p_hidden_dim
        self.p_ini_hidden_dim = p_ini_hidden_dim
        self.p_batch_size = p_batch_size
        self.p_output_dim = p_output_dim
        self.p_num_layers = p_num_layers
        self.p_rnn_type = p_rnn_type
        self.out_embed_d = out_embed_d
        self.outemb_d = outemb_d
        self.num_meta_path = num_meta_path
        # self.features = features
        self.content_dict = content_dict  # what is this? (Unused)

        # newly-added attention layer
        self.num_head = num_head
        self.attention_net = AttentionNet(self.num_meta_path, self.hidden_dim*2, self.hidden_dim*2, self.embed_d,
                                         self.num_head, att_num_layers, att_dropout)
        self.post_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.user_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)



        self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        # Define the initial linear hidden layer
        self.init_linear_text = nn.Linear(self.input_dim[0], self.ini_hidden_dim[0])
        self.init_linear_image = nn.Linear(self.input_dim[1], self.ini_hidden_dim[1])
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
        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.out_dropout = nn.Dropout(p=0.5)
        self.out_linear = nn.Linear(num_head*self.embed_d, self.outemb_d)
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight)
                # m.bias.data.fill_(0.1)

    # use Bi_RNN to aggregate node heterogeneous content
    # newly-added feature projection (project heterogeneous node content to the same feature space)[HetGAN]
    # feature projection is not used bc the dimensions of different node embedding are the same
    def feature_projection(self, node_id, node_type, emb_dict):
        input_a = []
        input_b = []
        if node_type == "post":
            input_a.append(emb_dict[node_id][0])  # text
            input_b.append(emb_dict[node_id][1])  # image
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_a)
            linear_input_image = self.init_linear_image(input_b)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_image = linear_input_image.view(linear_input_image.shape[0], 1, linear_input_image.shape[1])
            LSTM_out_text, _ = self.LSTM_text(linear_input_text)
            LSTM_out_image, _ = self.LSTM_image(linear_input_image)
            #LSTM_out_text = LSTM_out_text.squeeze(0)
            #LSTM_out_image = LSTM_out_image.squeeze(0)
            #print(LSTM_out_text.shape)
            concate = torch.cat((LSTM_out_text, LSTM_out_image), 1)
            mean_pooling = torch.mean(concate, 1)
            concate = self.post_projection(mean_pooling)

        if node_type == "user":
            input_a.append(emb_dict[node_id][1])  # user description
            input_b.append(emb_dict[node_id][0])  # other features
            input_a = torch.Tensor(input_a)
            input_b = torch.Tensor(input_b)
            linear_input_text = self.init_linear_text(input_a)
            linear_input_other = self.init_linear_other(input_b)
            linear_input_text = linear_input_text.view(linear_input_text.shape[0], 1, linear_input_text.shape[1])
            linear_input_other = linear_input_other.view(linear_input_other.shape[0], 1, linear_input_other.shape[1])
            LSTM_out_text, _ = self.LSTM_text(linear_input_text)
            LSTM_out_other, _ = self.LSTM_other(linear_input_other)
            #LSTM_out_text = LSTM_out_text.squeeze(0)
            #LSTM_out_other = LSTM_out_other.squeeze(0)
            concate = torch.cat((LSTM_out_text, LSTM_out_other), 1)
            mean_pooling = torch.mean(concate,1)
            concate = self.user_projection(mean_pooling)  # project to the same space
        return concate

    def Att_Aggregation(self, het_node, post_emb_dict, user_emb_dict):
        node_features = list()
        node_features.append(self.feature_projection(het_node.node_id, het_node.node_type, post_emb_dict))
        for id in het_node.neighbors_post:
            node_features.append(self.feature_projection(id, "post", post_emb_dict))
        for id in het_node.neighbors_user:
            node_features.append(self.feature_projection(id, "user", user_emb_dict))

        node_features = torch.stack(node_features, dim=0)

        res = self.attention_net([het_node.index_self_to_post, het_node.index_post_to_self,
                                    het_node.index_self_to_user, het_node.index_user_to_self],
                                 node_features)

        # res = global_mean_pool(res, batch=torch.LongTensor(np.arange(1)))
        # print(res.shape)



        return res[0]
    
    def output(self, c_embed_batch):
        batch_size = 1
        # make c_embed 3D tensor. Batch_size * 1 * embed_d
        c_embed = c_embed_batch.view(batch_size, 1, self.num_head*self.hidden_dim*2)
        c_embed = self.out_dropout(c_embed)
        c_embed_out = self.out_linear(c_embed)
        predictions = self.output_act(c_embed_out)  # log(1/(1+exp(-x)))    sigmoid = 1/(1+exp(-x))
        return predictions

    def forward(self, x):
        x = self.Att_Aggregation(het_node=x, post_emb_dict=post_emb_dict, user_emb_dict=user_emb_dict)
        #print("shape of x ", x.shape)
        x = self.output(c_embed_batch=x)
        return x


# In[42]:


# return a list of post nodes or user nodes.
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
        f = open(pathway + 'normalized_user_nodes_onehot.txt')
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


post_nodes = data_loader(
    pathway='/Users/jessica/Desktop/experiment_results/roberta/5_posts_20_users/normalized_post_nodes/',
    node_type="post")
user_nodes = data_loader(pathway='/Users/jessica/Desktop/experiment_results/roberta/5_posts_20_users/',
                         node_type="user")
post_emb_dict = {}
user_emb_dict = {}
for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb


# In[47]:


# split test set first
if path.exists('/Users/jessica/Desktop/experiment_results/test_index.txt'):
    train_val, test_set = load_train_test(post_nodes)
else:
    train_val, test_set = train_test(post_nodes, 4190, 465)

# Shuffle the order in post nodes
np.random.shuffle(train_val)

# K-fold validation index
train_index = []
val_index = []
num_folds = 9
kfold = KFold(num_folds, True, 1)
for train, val in kfold.split(train_val):
    train_index.append(train)
    val_index.append(val)

# best_models = []
# Initialize parameters
lr = 0.1 # lr = 0.8 at first
num_epoch = 40
batch_size = 8
PATH = '/Users/jessica/Desktop/experiment_results/roberta/5_posts_20_users/'
num_itr = 1
print('Start training')

for fold in range(num_itr):
    print("Start for fold", fold)
    best_val = 0.0
    running_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0
    best_epoch = 0
    net = Het_GNN(input_dim=[768, 512, 145], ini_hidden_dim=[200, 200, 150], hidden_dim=100, batch_size=1,
                  u_input_dim=200, u_hidden_dim=150, u_ini_hidden_dim=150,
                  u_batch_size=1, u_output_dim=200, u_num_layers=1, u_rnn_type='LSTM', p_input_dim=200,
                  p_hidden_dim=150, p_ini_hidden_dim=150, p_batch_size=1, p_output_dim=200, p_num_layers=1,
                  p_rnn_type='LSTM', out_embed_d=150, outemb_d=1)
    net.init_weights()
    print(net)
    # Set up optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)

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


# In[50]:


# Init net and optimizer skeletons
best_models = []
num_folds = 1
net = Het_GNN(input_dim=[768, 512, 145], ini_hidden_dim=[200, 200, 150], hidden_dim=100, batch_size=1,
                  u_input_dim=200, u_hidden_dim=150, u_ini_hidden_dim=150,
                  u_batch_size=1, u_output_dim=200, u_num_layers=1, u_rnn_type='LSTM', p_input_dim=200,
                  p_hidden_dim=150, p_ini_hidden_dim=150, p_batch_size=1, p_output_dim=200, p_num_layers=1,
                  p_rnn_type='LSTM', out_embed_d=200, outemb_d=1)
net.init_weights()
optimizer = optim.SGD(net.parameters(), lr=lr)

for count in range(num_folds):
    cur_PATH = PATH + 'best_model'+'_'+str(count)+'.tar'
    net, optimizer, epoch, best_val = load_checkpoint(net, optimizer, cur_PATH)
    print("best val is: ", best_val)
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

real_precision = real_true/(real_true+fake_count-fake_true)
fake_precision = fake_true/(fake_true+real_count-real_true)
real_recall = real_true/real_count
fake_recall = fake_true/fake_count
real_f1 = 2*real_precision*real_recall/(real_precision+real_recall)
fake_f1 = 2*fake_precision*fake_recall/(fake_precision+fake_recall)
print('Test loss: %.4f, Test accuracy: %.4f'% (test_loss/len(test_set), t/len(test_set)))
print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f'% (real_precision, real_recall, real_f1))
print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f'% (fake_precision, fake_recall, fake_f1))


# In[14]:





# In[ ]:




