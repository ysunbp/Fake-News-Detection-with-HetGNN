import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from os import path


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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

post_nodes = data_loader(pathway='F:/FYP_data/normalized_post_nodes/', node_type="post")
user_nodes = data_loader(pathway='F:/FYP_data/normalized_user_nodes/', node_type="user")
post_emb_dict = {}
user_emb_dict = {}
for user in user_nodes:
    user_emb_dict[user.node_id] = user.emb
for post in post_nodes:
    post_emb_dict[post.node_id] = post.emb


class Het_GNN(nn.Module):
    #features: list of HetNode class
    def __init__(self, input_dim, ini_hidden_dim, hidden_dim,
                 u_input_dim, u_hidden_dim, u_ini_hidden_dim, u_output_dim, u_num_layers,
                 p_input_dim, p_hidden_dim, p_ini_hidden_dim, p_output_dim, p_num_layers,
                 out_embed_d, outemb_d,
                 content_dict={}, num_layers=1, u_rnn_type='GRU', p_rnn_type='GRU', rnn_type='GRU', embed_d = 200):
        super(Het_GNN, self).__init__()
        self.input_dim = input_dim
        self.ini_hidden_dim = ini_hidden_dim
        self.hidden_dim = hidden_dim

        self.num_layers = num_layers
        self.embed_d = embed_d
        self.u_input_dim = u_input_dim
        self.u_hidden_dim = u_hidden_dim
        self.u_ini_hidden_dim = u_ini_hidden_dim

        self.u_output_dim = u_output_dim
        self.u_num_layers = u_num_layers
        self.u_rnn_type = u_rnn_type
        self.p_input_dim = p_input_dim
        self.p_hidden_dim = p_hidden_dim
        self.p_ini_hidden_dim = p_ini_hidden_dim

        self.p_output_dim = p_output_dim
        self.p_num_layers = p_num_layers
        self.p_rnn_type = p_rnn_type
        self.out_embed_d = out_embed_d
        self.outemb_d = outemb_d
        #self.features = features
        self.content_dict = content_dict
        self.p_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        self.u_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)
        # Define the initial linear hidden layer
        self.init_linear_text = nn.Linear(self.input_dim[0], self.ini_hidden_dim[0])
        self.init_linear_image = nn.Linear(self.input_dim[1], self.ini_hidden_dim[1])
        self.init_linear_other = nn.Linear(self.input_dim[2], self.ini_hidden_dim[2])
        # Define the GRU layer
        self.gru_text = eval('nn.' + rnn_type)(self.ini_hidden_dim[0], self.hidden_dim, self.num_layers, batch_first=True,
                                                bidirectional=True, dropout=0.4)
        self.gru_image = eval('nn.' + rnn_type)(self.ini_hidden_dim[1], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True, dropout=0.4)
        self.gru_other = eval('nn.' + rnn_type)(self.ini_hidden_dim[2], self.hidden_dim, self.num_layers, batch_first=True,
                                                 bidirectional=True, dropout=0.4)
        # Define same_type_agg
        self.u_init_linear = nn.Linear(self.u_input_dim, self.u_ini_hidden_dim)
        #self.u_init_dropout = nn.Dropout(p=0.2)
        self.u_gru = eval('nn.' + self.u_rnn_type)(self.u_ini_hidden_dim, self.u_hidden_dim, self.u_num_layers,
                                                    batch_first=True, bidirectional=True, dropout=0.4)
        self.u_linear = nn.Linear(self.u_hidden_dim * 2, self.u_output_dim)
        self.u_dropout = nn.Dropout(p=0.5)
        self.p_init_linear = nn.Linear(self.p_input_dim, self.p_ini_hidden_dim)
        #self.p_init_dropout = nn.Dropout(p=0.2)
        self.p_gru = eval('nn.' + self.p_rnn_type)(self.p_ini_hidden_dim, self.p_hidden_dim, self.p_num_layers,
                                                    batch_first=True, bidirectional=True, dropout=0.4)
        self.p_linear = nn.Linear(self.p_hidden_dim * 2, self.p_output_dim)
        self.p_dropout = nn.Dropout(p=0.5)
        self.act = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.out_dropout = nn.Dropout(p=0.25)
        self.out_linear = nn.Linear(self.out_embed_d, self.outemb_d)
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

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
            linear_input_text = linear_input_text.view(linear_input_text.shape[0],1,linear_input_text.shape[1])
            linear_input_image = linear_input_image.view(linear_input_image.shape[0],1,linear_input_image.shape[1])
            gru_out_text, self.hidden_text = self.gru_text(linear_input_text)
            gru_out_image, self.hidden_image = self.gru_image(linear_input_image)
            concate = torch.cat((gru_out_text, gru_out_image), 1)
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
            gru_out_text, self.hidden_text = self.gru_text(linear_input_text)
            gru_out_other, self.hidden_other = self.gru_other(linear_input_other)
            concate = torch.cat((gru_out_text, gru_out_other), 1)

        # mean pooling all the states
        mean_pooling = torch.mean(concate, 1)

        for i in neighbor_id:
            if ("post", i) in self.content_dict:
                mean_pooling = torch.cat(mean_pooling, self.content_dict[i], dim=0)
        for i in range(len(new_id)):
            self.content_dict[i] = mean_pooling[i]
        return mean_pooling

    #features: list of [(id)]
    def SameType_Agg_Bi_RNN(self, neighbor_id, node_type):
        content_embedings = self.Bi_RNN(neighbor_id, node_type, post_emb_dict, user_emb_dict)
        if node_type == 'post':
            linear_input = self.p_init_linear(content_embedings)
            #linear_input = self.p_init_dropout(linear_input)
            linear_input = linear_input.view(linear_input.shape[0],1,linear_input.shape[1])
            gru_out, hidden = self.p_gru(linear_input)
            last_state = self.p_linear(gru_out)
            last_state = self.p_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling
        else:
            linear_input = self.u_init_linear(content_embedings)
            #linear_input = self.u_init_dropout(linear_input)
            linear_input = linear_input.view(linear_input.shape[0], 1, linear_input.shape[1])
            gru_out, hidden = self.u_gru(linear_input)
            last_state = self.u_linear(gru_out)
            last_state = self.u_dropout(last_state)
            mean_pooling = torch.mean(last_state, 0)
            return mean_pooling

    def node_het_agg(self, het_node): #heterogeneous neighbor aggregation

        #attention module
        c_agg_batch = self.Bi_RNN([het_node.node_id], het_node.node_type, post_emb_dict, user_emb_dict)
        u_agg_batch = self.SameType_Agg_Bi_RNN(het_node.neighbors_user, "user")
        p_agg_batch = self.SameType_Agg_Bi_RNN(het_node.neighbors_post, "post")

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
        #print(atten_w)

        #weighted combination
        concate_embed = torch.cat((c_agg_batch, u_agg_batch, p_agg_batch), 1).view(len(c_agg_batch), 3, self.embed_d)
        weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

        return weight_agg_batch
    
    def output(self, c_embed_batch):

        batch_size = 1
        # make c_embed 3D tensor. Batch_size * 1 * embed_d
        c_embed = c_embed_batch.view(batch_size, 1, self.out_embed_d)
        c_embed = self.out_dropout(c_embed)
        c_embed_out = self.out_linear(c_embed)
        predictions = self.output_act(c_embed_out) #log(1/(1+exp(-x)))    sigmoid = 1/(1+exp(-x))
        return predictions

    def forward(self, x):
        #print(x.shape)
        x = self.node_het_agg(het_node = x)
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
    y = range(train_size+test_size)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = test_size, random_state = 42)
    np.savetxt('F:\\FYP_models\\5_folds\\train_index.txt', y_train)
    np.savetxt('F:\\FYP_models\\5_folds\\test_index.txt', y_test)
    return X_train, X_test

def load_train_test(data, test_index_path = 'F:\\FYP_models\\5_folds\\test_index.txt'):
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

if path.exists('F:\\FYP_models\\5_folds\\test_index.txt'):
    train_val, test_set = load_train_test(post_nodes)
else:
    train_val, test_set = train_test(post_nodes, 4190, 465)

#train_val = post_nodes[:4190]
#test_set = post_nodes[4190:]



# Shuffle the order in post nodes
np.random.shuffle(train_val)

# K-fold validation index
train_index = []
val_index = []
num_splits = 9
kfold = KFold(num_splits, True, 1)
for train, val in kfold.split(train_val):
    train_index.append(train)
    val_index.append(val)


#best_models = []
# Initialize parameters
lr = 0.06
num_epoch = 30
num_folds = 5
PATH = 'F:/FYP_models/batch/'

print('Start training')


for fold in range(num_folds):
    print("Start for fold", fold+1)
    best_val = 100
    running_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0
    best_epoch = 0
    net = Het_GNN(input_dim = [300, 512, 12], ini_hidden_dim = [100, 100, 100], hidden_dim=100, u_input_dim=200, u_hidden_dim=100, u_ini_hidden_dim=100,
                                   u_output_dim=200, u_num_layers=1, u_rnn_type='GRU', p_input_dim=200,
                                   p_hidden_dim=100, p_ini_hidden_dim=100, p_output_dim=200, p_num_layers=1,
                                    p_rnn_type='GRU',out_embed_d=200, outemb_d=1)
    net.init_weights()
    print(net)
    # Set up optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.25)

    for epoch in range(num_epoch):
        print('Epoch:', epoch+1)
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
        for i in range(len(train_set)):
            optimizer.zero_grad()
            output = net(train_set[i])
            if (output.item() >= 0.5 and train_set[i].label == 1) or (output.item() < 0.5 and train_set[i].label == 0):
                c += 1
            loss = BCELoss(predictions=output, true_label=train_set[i].label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('Fold: %d, Epoch: %d, step: %5d, loss: %.4f, acc: %.4f'%
                      (fold + 1, epoch + 1, i + 1, running_loss / 100, c/100))
                running_loss = 0.0
                c = 0.0
        for j in range(len(val_set)):
            output = net(val_set[j])
            if val_set[j].label == 1:
                real_count += 1
                if output.item()>=0.5:
                    real_true += 1
            else:
                fake_count += 1
                if output.item()<0.5:
                    fake_true += 1
            if (output.item() >= 0.5 and val_set[j].label == 1) or (output.item() < 0.5 and val_set[j].label == 0):
                v += 1
            vloss = BCELoss(predictions=output, true_label=val_set[j].label)
            val_loss += vloss.item()
        val_acc = v/len(val_set)
        real_precision = real_true/(real_true+fake_count-fake_true)
        fake_precision = fake_true/(fake_true+real_count-real_true)
        real_recall = real_true/real_count
        fake_recall = fake_true/fake_count
        real_f1 = 2*real_precision*real_recall/(real_precision+real_recall)
        fake_f1 = 2*fake_precision*fake_recall/(fake_precision+fake_recall)
        print('Validation loss: %.4f, Validation accuracy: %.4f'% (val_loss/len(val_set), val_acc))
        print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f'% (real_precision, real_recall, real_f1))
        print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f'% (fake_precision, fake_recall, fake_f1))
        if val_loss/len(val_set) < best_val:
            print('Update model at epoch:', epoch+1)
            cur_PATH = PATH + 'best_model'+'_'+str(fold)+'.tar'
            save_checkpoint(net, optimizer, cur_PATH, epoch+1, val_acc)
            best_val = val_loss/len(val_set)
            best_epoch = epoch
        scheduler.step()
        if epoch - best_epoch >= 5:
            break

    print('Finish training')


print('==============================================================')

# Init net and optimizer skeletons
best_models = []
num_folds = 5
net = Het_GNN(input_dim = [300, 512, 12], ini_hidden_dim = [100, 100, 100], hidden_dim=100, u_input_dim=200, u_hidden_dim=100, u_ini_hidden_dim=100,
                                   u_output_dim=200, u_num_layers=1, u_rnn_type='GRU', p_input_dim=200,
                                   p_hidden_dim=100, p_ini_hidden_dim=100, p_output_dim=200, p_num_layers=1,
                                    p_rnn_type='GRU',out_embed_d=200, outemb_d=1)
net.init_weights()
optimizer = optim.SGD(net.parameters(), lr=lr)

for count in range(num_folds):
    cur_PATH = PATH + 'best_model'+'_'+str(count)+'.tar'
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

real_precision = real_true/(real_true+fake_count-fake_true)
fake_precision = fake_true/(fake_true+real_count-real_true)
real_recall = real_true/real_count
fake_recall = fake_true/fake_count
real_f1 = 2*real_precision*real_recall/(real_precision+real_recall)
fake_f1 = 2*fake_precision*fake_recall/(fake_precision+fake_recall)
print('Test loss: %.4f, Test accuracy: %.4f'% (test_loss/len(test_set), t/len(test_set)))
print('Real Precision: %.4f, Real Recall: %.4f, Real F1: %.4f'% (real_precision, real_recall, real_f1))
print('Fake Precision: %.4f, Fake Recall: %.4f, Fake F1: %.4f'% (fake_precision, fake_recall, fake_f1))


