import random
from numpy.random import choice
from tqdm import tqdm
import numpy as np
import os

def softmax(x: list):
    return np.exp(x) / np.sum(np.exp(x))


def read_graph():
    """
    NOTE: A user's user neighbor list may contain duplication. It is
    intentionally kept because the more posts in common 2 users share,
    the more similar they are.
    """

    p_adj_list, u_adj_list = dict(), dict()

    with open(data_path + post_user_f, "r") as f:
        for l in tqdm(f.readlines(), desc='Read user-post edges'):
            pid, uid = l.strip().split(": ")
            pid, uid = 'p' + pid, 'u' + uid
            if pid not in p_adj_list.keys():
                p_adj_list[pid] = {'p': {'id': [], 'prob': []}, 'u': []}
            if uid not in u_adj_list.keys():
                u_adj_list[uid] = {'p': [], 'u': {'id': [], 'prob': []}}
            p_adj_list[pid]['u'].append(uid)  # user neighbor of post
            u_adj_list[uid]['p'].append(pid)  # post neighbor of user

    with open(post_weight_path, 'r') as f:
        for l in tqdm(f.readlines(), desc='Read post-post edges'):
            l = l.strip().split()
            pid1 = 'p' + l[0]
            for ll in l[1:]:
                pid2, score_str = ll.split(':')
                p_adj_list[pid1]['p']['id'].append(
                    'p' + pid2)  # post neighbor of post
                p_adj_list[pid1]['p']['prob'].append(
                    float(score_str))  # post neighbor of post
            p_adj_list[pid1]['p']['prob'] = softmax(
                p_adj_list[pid1]['p']['prob'])

    user_dir_list = os.listdir(user_weight_dir)
    for fname in tqdm(user_dir_list, desc='Read user-user edges'):
        with open(os.path.join(user_weight_dir, fname), 'r') as f:
            for l in f.readlines():
                l = l.strip().split()
                uid1 = 'u' + l[0]
                if uid1 not in u_adj_list.keys():
                    u_adj_list[uid1] = {'p': [], 'u': {'id': [], 'prob': []}}
                for ll in l[1:]:
                    uid2, score_str = ll.split(':')
                    u_adj_list[uid1]['u']['id'].append(
                        'u' + uid2)  # user neighbor of user
                    u_adj_list[uid1]['u']['prob'].append(
                        float(score_str))  # user neighbor of user
    for uid1 in tqdm(u_adj_list.keys(), 'Softmax u-u prob'):
        u_adj_list[uid1]['u']['prob'] = softmax(
            u_adj_list[uid1]['u']['prob'])
    
    # with open(user_user_f, 'r') as f:
    #     for l in tqdm(f.readlines(), desc='Read user-user edges'):
    #         uid1, uids_str = l.strip().split(': ')
    #         uid1 = 'u' + uid1
    #         uid2s = ['u' + uid2 for uid2 in uids_str.split(', ')]
    #         for uid2 in uid2s:
    #             if uid1 not in u_adj_list:
    #                 u_adj_list[uid1] = {'p' : [], 'u' : []}
    #             if uid2 not in u_adj_list:
    #                 u_adj_list[uid2] = {'p' : [], 'u' : []}
    #             u_adj_list[uid1]['u'].append(uid2)  # post neighbor of user
    #             u_adj_list[uid2]['u'].append(uid1)  # post neighbor of user

    return p_adj_list, u_adj_list

def _select_neighbors(start_node, u_neighbors, p_neighbors, p_neigh_list_, max_uniq_neigh_u_test, max_uniq_neigh_p_test):
    def get_top_k_most_frequent(neighbors, k):
        counter = dict()
        for node in neighbors:
            if node not in counter:
                counter[node] = 0
            counter[node] += 1
        items = sorted(list(counter.items()), key=lambda x: -x[1])
        top_k = [items[i][0] for i in range(min(k, len(items)))]
        return top_k
    top_k_u = get_top_k_most_frequent(u_neighbors, max_uniq_neigh_u_test)
    top_k_p = get_top_k_most_frequent(p_neighbors, max_uniq_neigh_p_test + 1)[
        1:
    ]  # hack: exclude self
    p_neigh_list_[start_node] = {"u": top_k_u, "p": top_k_p}

def save_result(p_neigh_list, max_uniq_neigh_p_test, max_uniq_neigh_u_test):
    print('Save result settings:', max_uniq_neigh_p_test, max_uniq_neigh_u_test)
    mini_p_neigh_list = dict()
    u_involved = set()
    for node, neighs in tqdm(p_neigh_list.items(), 'select neighbors'):
        _select_neighbors(node, neighs['u'], neighs['p'], mini_p_neigh_list, max_uniq_neigh_u_test, max_uniq_neigh_p_test)
        u_involved = u_involved.union(mini_p_neigh_list[node]['u'])
    
    with open(data_path + post_neigh_f, "w") as f:
        print("Writing {} posts' neighbors.".format(len(mini_p_neigh_list)))
        f.writelines(
            [
                "{}: {} {}\n".format(node, " ".join(
                    neighs["p"]), " ".join(neighs["u"]))
                for node, neighs in mini_p_neigh_list.items()
            ]
        )
    with open(data_path + users_involved_f, "w") as f:
        u_involved = list(u_involved)
        print("Writing {} users involved.".format(len(u_involved)))
        f.write(" ".join(u_involved) + "\n")

    n_nodes = len(mini_p_neigh_list.items())
    p_lens = sum([len(neighs["p"]) for _, neighs in mini_p_neigh_list.items()])
    u_lens = sum([len(neighs["u"]) for _, neighs in mini_p_neigh_list.items()])
    with open(data_path + stats_f, 'w') as f:
        f.write('avg p_len: {:.8f}\navg u_len: {:.8f}\n'.format(p_lens / n_nodes, u_lens / n_nodes))

def random_walk_with_restart(
    restart_rate,
    min_neigh_u,
    min_neigh_p,
    num_neigh_to_record,
    pp_rate,
    uu_rate,
    max_uniq_neigh_u, 
    max_uniq_neigh_p
):
    p_neigh_list = dict()

    def add_neighbor(cur_node):
        if cur_node[0] == "u":
            if len(u_neighbors) >= min_neigh_u and len(
                    p_neighbors) < min_neigh_p:
                return
            u_neighbors.append(cur_node)
            u_neigh_uniq.add(cur_node)
        else:  # cur_node[0] == 'p'
            if len(p_neighbors) >= min_neigh_p and len(
                    u_neighbors) < min_neigh_u:
                return
            p_neighbors.append(cur_node)
            p_neigh_uniq.add(cur_node)

    print("Each node takes turns to be the starting node...")

    for start_node in tqdm(list(p_adj_list.keys())):
        u_neighbors, p_neighbors = [], []
        u_neigh_uniq, p_neigh_uniq = set(), set()
        cur_node = start_node
        while len(u_neighbors) + len(p_neighbors) < num_neigh_to_record:
            rand_p = random.random()  # return p
            if rand_p < restart_rate:
                cur_node = start_node
            else:
                if cur_node[0] == "p":
                    if random.random() < pp_rate:
                        cur_node = choice(
                            p_adj_list[cur_node]['p']['id'], 1, p=p_adj_list[cur_node]['p']['prob'])[0]
                    else:
                        cur_node = random.choice(p_adj_list[cur_node]['u'])
                    add_neighbor(cur_node)
                elif cur_node[0] == "u":
                    if random.random() < uu_rate:
                        cur_node = choice(
                            u_adj_list[cur_node]['u']['id'], 1, p=u_adj_list[cur_node]['u']['prob'])[0]
                    else:
                        cur_node = random.choice(u_adj_list[cur_node]['p'])
                    add_neighbor(cur_node)
        _select_neighbors(start_node, u_neighbors, p_neighbors, p_neigh_list, max_uniq_neigh_u, max_uniq_neigh_p)
    return p_neigh_list


if __name__ == "__main__":

    data_path = "/rwproject/kdd-db/20-rayw1/fyp_code/"
    post_user_f = "tweet_user.txt"
    # user_user_f = 'user_user.txt'
    user_weight_dir = '/rwproject/kdd-db/20-rayw1/data/edge_weight_user'
    post_weight_path = '/rwproject/kdd-db/20-rayw1/data/edge_weight_post.txt'

    p_adj_list, u_adj_list = read_graph()

    p_u_tests = [
        (2, 2),
        (5, 5),
        (7, 7),
        (10, 10),
        (12, 12),
        (15, 15),
        (5, 2),
        (5, 7),
        (5, 12),
        (5, 20),
        (10, 2),
        (10, 7),
        (10, 12),
        (10, 20),
        (15, 2),
        (15, 7),
        (15, 12),
        (15, 20),
    ]

    # for neighbor selection
    max_uniq_neigh_p = max([e[0] for e in p_u_tests])
    max_uniq_neigh_u = max([e[1] for e in p_u_tests])

    p_neigh_list = random_walk_with_restart(
        restart_rate = 0.5,
        min_neigh_u = 300,
        min_neigh_p = 500,
        num_neigh_to_record = 1000,
        pp_rate = 0.5,
        uu_rate = 0.5,
        max_uniq_neigh_u = max_uniq_neigh_u,
        max_uniq_neigh_p = max_uniq_neigh_p,
    )

    for max_uniq_neigh_p_test, max_uniq_neigh_u_test in p_u_tests:
        configuration_tag = f"weighted_{max_uniq_neigh_p_test}_posts_{max_uniq_neigh_u_test}_users"
        output_dir = f"rwr_results/{configuration_tag}/"
        post_neigh_f = output_dir + "post_neighbors.txt"
        users_involved_f = output_dir + "users_involved.txt"
        stats_f = output_dir + 'stats.txt'

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        print("#" * 15 + ' ' + configuration_tag + ' ' + "#" * 15)
        save_result(p_neigh_list, max_uniq_neigh_p_test, max_uniq_neigh_u_test)
