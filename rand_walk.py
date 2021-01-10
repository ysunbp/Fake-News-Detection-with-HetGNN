import random
from tqdm import tqdm
import os

data_path = "/rwproject/kdd-db/20-rayw1/fyp_code/"
# input
post_user_f = "tweet_user.txt"
user_post_f = "user_tweet.txt"
user_user_f = "user_user.txt"


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
    p_adj_list, u_adj_list = dict(), dict()
    p_neigh_list = dict()
    u_involved = set()

    print("Read the graph...")
    for adj_f, adj_list, main_type, neigh_type in [
        (post_user_f, p_adj_list, "p", "u"),
        (user_post_f, u_adj_list, "u", "p"),
        (user_user_f, u_adj_list, "u", "u"),
    ]:
        with open(data_path + adj_f, "r") as f:
            for l in f.readlines():
                l = l.strip().split(": ")
                node = main_type + l[0]
                if node not in adj_list:
                    adj_list[node] = []
                adj_list[node].extend([neigh_type + i for i in l[1].split(", ")])

    def add_neighbor(cur_node):
        if cur_node[0] == "u":
            if len(u_neighbors) >= min_neigh_u and len(p_neighbors) < min_neigh_p:
                return
            u_neighbors.append(cur_node)
            u_neigh_uniq.add(cur_node)
        else:  # cur_node[0] == 'p'
            if len(p_neighbors) >= min_neigh_p and len(u_neighbors) < min_neigh_u:
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
                        cur_node = random.choice(p_adj_list[cur_node])
                        add_neighbor(cur_node)
                    elif cur_node[0] == "u":
                        cur_node = random.choice(u_adj_list[cur_node])
                        add_neighbor(cur_node)
            write_neighbor(start_node)
            u_involved = u_involved.union(neigh_list[start_node]["u"])

    print("Save the result...")
    with open(data_path + post_neigh_f, "w") as f:
        print("Writing {} posts' neighbors.".format(len(p_neigh_list)))
        f.writelines(
            [
                "{}: {} {}\n".format(node, " ".join(neighs["p"]), " ".join(neighs["u"]))
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
    min_neigh_u = 300
    min_neigh_p = 500
    num_neigh_to_record = 1000

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
    # max_uniq_neigh_p = 5
    # max_uniq_neigh_u = 20

    for max_uniq_neigh_p, max_uniq_neigh_u in p_u_tests:

        # output
        configuration_tag = f"{max_uniq_neigh_p}_posts_{max_uniq_neigh_u}_users"
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
