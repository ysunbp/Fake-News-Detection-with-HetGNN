"""
To avoid being killed, process graphs one by one. Deduplicate afterwards.
To avoid bugs, all ids must be int. Check it.

User-user edge weights file sizes sum to 333G, which can't be read at once
in the 30G RAM obviously. => Solution:
1. Streamlined, e.g. graph generator by networkX
2. Set cutoff to reduce edges
"""
import json
import os
import random
from multiprocessing import Process
from typing import Dict

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm


class Node():
    def __init__(self, id: int, vec):
        self.id = id
        self.vec = vec
        self.neighbors = set()


def _process_nodes(nodes: Dict[int, Node], process_number=0):
    def _cosine_similarity(a, b):
        cos_sim = np.dot(a, b) / (norm(a) * norm(b))
        return cos_sim
    lines = []
    generator = tqdm(
        nodes.items(),
        desc='process nodes') if process_number == 0 else nodes.items()
    for id, node in generator:
        neigh_list = [(nid, _cosine_similarity(node.vec, nodes[nid].vec))
                      for nid in node.neighbors]
        neigh_list = sorted(neigh_list, key=lambda x: -x[1])
        neigh_str = ' '.join(
            [f'{uid}:{score:.6f}' for uid, score in neigh_list])
        lines.append(f'{id} {neigh_str}\n')
    return lines


def _process_some_user_files(
        dir_list,
        finished_dir_list,
        mean,
        std,
        process_number):
    def _read_users(filename):
        with open(os.path.join(user_features_dir, filename), 'r') as fin:
            lines = fin.readlines()
        
        segs = lines[0].strip().split()
        id = int(segs[0])
        vec = np.array([float(i) for i in segs[1:]])
        vec = ((vec - mean) / std) if standardize else vec
        author = Node(id, vec)

        users = dict()
        generator = tqdm(
            lines[1:], desc='read users') if process_number == 0 else lines[1:]
        for line in generator:
            segs = line.strip().split()
            id = int(segs[0])
            if id in users.keys():
                continue  # duplicates exist
            vec = np.array([float(i) for i in segs[1:]])
            vec = ((vec - mean) / std) if standardize else vec
            users[id] = Node(id, vec)
        author.neighbors = set([u2 for u2 in users.keys() if author.id != u2])
        for u2 in users.keys():
            users[u2].neighbors = {author.id}
        users[author.id] = author
        return users

    sorted_dir_list = [(fname, len(open(os.path.join(user_features_dir, fname), 'r').readlines()))
                       for fname in dir_list if fname not in finished_dir_list]
    sorted_dir_list = sorted(sorted_dir_list, key=lambda x: x[1])

    for i, (filename, _) in enumerate(sorted_dir_list):
        if process_number == 0:
            print('  users processed: {} over {} ({:.1f}%)'.format(
                i + 1, len(dir_list), (i + 1) / len(dir_list) * 100))
        users = _read_users(filename)
        lines = _process_nodes(users, process_number)
        with open(os.path.join(user_nodes_out_dir, filename), 'w') as fout:
            fout.writelines(lines)


def process_users(n_processes=1):
    def _get_user_stats():
        if os.path.isfile('user_stats.json'):
            stats = json.load(open('user_stats.json', 'r'))
            mean, std = np.array(stats['mean']), np.array(stats['std'])
        else:
            vecs = []
            for i, filename in enumerate(
                    tqdm(os.listdir(user_features_dir), desc=f'get user stats')):
                with open(os.path.join(user_features_dir, filename), 'r') as fin:
                    for line in fin.readlines():
                        vecs.append(np.array([float(i)
                                              for i in line.strip().split()[1:]]))
            mat = np.stack(vecs)
            mean = np.mean(mat, axis=0)
            std = np.std(mat, axis=0)
            json.dump({'mean': mean.tolist(), 'std': std.tolist()},
                      open('user_stats.json', 'w'))
        return mean, std

    (mean, std) = _get_user_stats() if standardize else (None, None)
    dir_list = os.listdir(user_features_dir)
    if not os.path.isdir(user_nodes_out_dir):
        os.mkdir(user_nodes_out_dir)
    # finished_dir_list = os.listdir(user_nodes_out_dir)
    finished_dir_list = []

    random.shuffle(dir_list)
    n_files = (len(dir_list) + n_processes - 1) // n_processes
    ps = []
    for i in range(n_processes):
        fnames = dir_list[i * n_files: min((i + 1) * n_files, len(dir_list))]
        p = Process(target=_process_some_user_files,
                    args=(fnames, finished_dir_list, mean, std, i))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    print("all users processed")


def process_posts():
    def _get_post_stats():
        vecs = []
        for i, filename in enumerate(
                tqdm(os.listdir(post_features_dir), desc=f'get post stats')):
            with open(os.path.join(post_features_dir, filename), 'r') as fin:
                vecs.append(
                    np.array([float(i) for i in fin.readlines()[0].strip().split()]))
        mat = np.stack(vecs)
        mean = np.mean(mat, axis=0)
        std = np.std(mat, axis=0)
        return mean, std

    mean, std = _get_post_stats() if standardize else 0, 1
    posts = dict()
    for i, filename in enumerate(
            tqdm(os.listdir(post_features_dir), desc=f'read posts')):
        # if i == 10:
        #     break
        with open(os.path.join(post_features_dir, filename), 'r') as fin:
            id = int(filename[:-4])
            for line in fin.readlines():
                vec = np.array([float(i) for i in line.strip().split()])
                vec = (vec - mean) / std if standardize else vec
                posts[id] = Node(id, vec)
    for u1 in tqdm(posts.keys(), desc='build post graph'):
        posts[u1].neighbors = set([u2 for u2 in posts.keys() if u1 != u2])
    lines = _process_nodes(posts)
    with open(post_nodes_out_path, 'w') as fout:
        fout.writelines(lines)


if __name__ == '__main__':
    standardize = True
    user_features_dir = '/rwproject/kdd-db/20-rayw1/rumdect/weibo_user_feature'
    post_features_dir = '/rwproject/kdd-db/20-rayw1/data/weibo/xlm-roberta-base/posts'
    user_nodes_out_dir = '/rwproject/kdd-db/20-rayw1/data/edge_weight_user'
    post_nodes_out_path = '/rwproject/kdd-db/20-rayw1/data/edge_weight_post.txt'

    # process_posts()
    process_users(n_processes=8)
