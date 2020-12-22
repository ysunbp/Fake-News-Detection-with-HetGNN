import random
from tqdm import tqdm

data_path = '/rwproject/kdd-db/20-rayw1/fyp_code/'
# input
tweet_user_f = 'tweet_user.txt'
user_tweet_f = 'user_tweet.txt'
user_user_f = 'user_user.txt'
# output
het_neigh_f = 'het_neigh.txt'

restart_rate = 0.5
max_neigh_size = 80 #有個node會卡在92 不知為何
neigh_type_size_constraint = 46  # max num of same type neighbors

t_adj_list, u_adj_list = dict(), dict()
t_neigh_list, u_neigh_list = dict(), dict()

print("Read the graph...")
for adj_f, adj_list, main_type, neigh_type in [
    (tweet_user_f, t_adj_list, 't', 'u'), 
    (user_tweet_f, u_adj_list, 'u', 't'), 
    (user_user_f, u_adj_list, 'u', 'u'), 
    ]:
    with open(data_path + adj_f, 'r') as f:
        for l in f.readlines():
            l = l.strip().split(': ')
            node = main_type + l[0]
            if node not in adj_list:
                adj_list[node] = []
            adj_list[node].extend([neigh_type + i for i in l[1].split(', ')])
    
print("Each node takes turns to be the starting node...")
lists = [(t_adj_list, t_neigh_list), (u_adj_list, u_neigh_list)]
for adj_list, neigh_list in lists:
    for start_node in tqdm(list(adj_list.keys())):
        neigh_L, u_L, t_L = 0, 0, 0
        neigh_list[start_node] = []
        cur_node = start_node
        while neigh_L < max_neigh_size:
            # print(start_node, neigh_L)
            rand_p = random.random() #return p
            if rand_p < restart_rate:
                cur_node = start_node
            else:
                if cur_node[0] == 't':
                    cur_node = random.choice(t_adj_list[cur_node])
                    if u_L < neigh_type_size_constraint:
                        neigh_list[start_node].append(cur_node)
                        neigh_L += 1
                        u_L += 1
                elif cur_node[0] == 'u':
                    cur_node = random.choice(u_adj_list[cur_node])
                    if t_L < neigh_type_size_constraint:
                        neigh_list[start_node].append(cur_node)
                        neigh_L += 1
                        t_L += 1

print("Save the result...")
with open(data_path + het_neigh_f, 'w') as f:
    for neigh_list in [t_neigh_list, u_neigh_list]:
        f.writelines([node + ':' + ','.join(neighs) + '\n' for node, neighs in neigh_list.items()])