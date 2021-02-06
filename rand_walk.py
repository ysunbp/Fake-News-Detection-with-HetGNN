import random
from numpy.random import choice
from tqdm import tqdm
import numpy as np
import os

data_path = "/rwproject/kdd-db/20-rayw1/fyp_code/"
# input
post_user_f = "tweet_user.txt"
# user_user_f = "user_user.txt"
user_weight_dir = '/rwproject/kdd-db/20-rayw1/data/edge_weight_user'
post_weight_path = '/rwproject/kdd-db/20-rayw1/data/edge_weight_post.txt'

pp_rate = 0.5
uu_rate = 0.5


def softmax(x: list):
    return np.exp(x) / np.sum(np.exp(x))


def read_graph():
    """
    NOTE: A user's user neighbor list may contain duplication. It is
    intentionally kept because the more posts in common 2 users share,
    the more similar they are.
    """

    p_adj_list, u_adj_list = dict(), dict()

    print("Reading user-post edges...")
    with open(data_path + post_user_f, "r") as f:
        for l in f.readlines():
            pid, uid = l.strip().split(": ")
            pid, uid = 'p' + pid, 'u' + uid
            if pid not in p_adj_list.keys():
                p_adj_list[pid] = {'p': {'id': [], 'probability': []}, 'u': []}
            if uid not in u_adj_list.keys():
                u_adj_list[uid] = {'p': [], 'u': {'id': [], 'probability': []}}
            p_adj_list[pid]['u'].append(uid)  # user neighbor of post
            u_adj_list[uid]['p'].append(pid)  # post neighbor of user

    print("Reading post-post edges...")
    with open(post_weight_path, 'r') as f:
        for l in f.readlines():
            l = l.strip().split()
            pid1 = 'p' + l[0]
            for ll in l[1:]:
                pid2, score_str = ll.split(':')
                p_adj_list[pid1]['p']['id'].append(
                    'p' + pid2)  # post neighbor of post
                p_adj_list[pid1]['p']['probability'].append(
                    float(score_str))  # post neighbor of post
            p_adj_list[pid1]['p']['probability'] = softmax(
                p_adj_list[pid1]['p']['probability'])

    print("Reading user-user edges...")
    user_dir_list = os.listdir(user_weight_dir)
    for fname in tqdm(user_dir_list, desc='read u-u edges'):
        with open(os.path.join(user_weight_dir, fname), 'r') as f:
            for l in f.readlines():
                l = l.strip().split()
                uid1 = 'p' + l[0]
                if uid1 not in u_adj_list.keys():
                    u_adj_list[uid1] = {'p': [], 'u': {
                        'id': [], 'probability': []}}
                for ll in l[1:]:
                    uid2, score_str = ll.split(':')
                    u_adj_list[uid1]['u']['id'].append(
                        'u' + uid2)  # user neighbor of user
                    u_adj_list[uid1]['u']['probability'].append(
                        float(score_str))  # user neighbor of user
    for uid1 in tqdm(u_adj_list.keys(), 'softmaxing'):
        u_adj_list[uid1]['u']['probability'] = softmax(
            u_adj_list[uid1]['u']['probability'])

    return p_adj_list, u_adj_list


def random_walk_with_restart(
    restart_rate,
    min_neigh_u,
    min_neigh_p,
    num_neigh_to_record,
    max_uniq_neigh_u,
    max_uniq_neigh_p,
    post_neigh_f,
    users_involved_f,
):
    p_adj_list, u_adj_list = read_graph()
    p_neigh_list = dict()
    u_involved = set()

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

    def get_top_k_most_frequent(neighbors, k):
        counter = dict()
        for node in neighbors:
            if node not in counter:
                counter[node] = 0
            counter[node] += 1
        items = sorted(list(counter.items()), key=lambda x: -x[1])
        top_k = [items[i][0] for i in range(min(k, len(items)))]
        return top_k

    def write_neighbor(start_node):
        top_k_u = get_top_k_most_frequent(u_neighbors, max_uniq_neigh_u)
        top_k_p = get_top_k_most_frequent(p_neighbors, max_uniq_neigh_p + 1)[
            1:
        ]  # hack: exclude self
        top_k_u_len.append(len(top_k_u))
        top_k_p_len.append(len(top_k_p))
        neigh_list[start_node] = {"u": top_k_u, "p": top_k_p}

    print("Each node takes turns to be the starting node...")
    top_k_u_len = []
    top_k_p_len = []
    lists = [
        (p_adj_list, p_neigh_list),
    ]
    for adj_list, neigh_list in lists:
        for start_node in tqdm(list(adj_list.keys())):
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
                                p_adj_list[cur_node]['p']['id'], 1, p=p_adj_list[cur_node]['p']['probability'])
                        else:
                            cur_node = random.choice(p_adj_list[cur_node]['u'])
                        add_neighbor(cur_node)
                    elif cur_node[0] == "u":
                        if random.random() < uu_rate:
                            cur_node = choice(
                                u_adj_list[cur_node]['u']['id'], 1, p=u_adj_list[cur_node]['u']['probability'])
                        else:
                            cur_node = random.choice(u_adj_list[cur_node])
                        add_neighbor(cur_node)
            write_neighbor(start_node)
            u_involved = u_involved.union(neigh_list[start_node]["u"])

    print("Save the result...")
    with open(data_path + post_neigh_f, "w") as f:
        print("Writing {} posts' neighbors.".format(len(p_neigh_list)))
        f.writelines(
            [
                "{}: {} {}\n".format(node, " ".join(
                    neighs["p"]), " ".join(neighs["u"]))
                for node, neighs in p_neigh_list.items()
            ]
        )
    with open(data_path + users_involved_f, "w") as f:
        u_involved = list(u_involved)
        print("Writing {} users involved.".format(len(u_involved)))
        f.write(" ".join(u_involved) + "\n")

    print(
        "Top-k post neighrbors: Mean of k = {}".format(
            sum(top_k_p_len) / len(top_k_p_len)
        )
    )
    print(
        "Top-k user neighrbors: Mean of k = {}".format(
            sum(top_k_u_len) / len(top_k_u_len)
        )
    )


if __name__ == "__main__":
    # for random walk with restart
    restart_rate = 0.5
    min_neigh_u = 3
    min_neigh_p = 5
    num_neigh_to_record = 10

    p_u_tests = [
        (2, 2),
        # (5, 5),
        # (7, 7),
        # (10, 10),r
        # (12, 12),
        # (15, 15),
        # (5, 2),
        # (5, 7),
        # (5, 12),
        # (5, 20),
        # (10, 2),
        # (10, 7),
        # (10, 12),
        # (10, 20),
        # (15, 2),
        # (15, 7),
        # (15, 12),
        # (15, 20),
    ]

    # for neighbor selection
    # max_uniq_neigh_p = 5
    # max_uniq_neigh_u = 20

    for max_uniq_neigh_p, max_uniq_neigh_u in p_u_tests:

        # output
        configuration_tag = f"weighted_{max_uniq_neigh_p}_posts_{max_uniq_neigh_u}_users"
        output_dir = f"random_walk_results/{configuration_tag}/"
        post_neigh_f = output_dir + "post_neighbors.txt"
        users_involved_f = output_dir + "users_involved.txt"

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        print("#" * 15 + configuration_tag + "#" * 15)
        random_walk_with_restart(
            restart_rate,
            min_neigh_u,
            min_neigh_p,
            num_neigh_to_record,
            max_uniq_neigh_u,
            max_uniq_neigh_p,
            post_neigh_f,
            users_involved_f,
        )
