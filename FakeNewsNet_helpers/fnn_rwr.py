"""
* All edges are undirected.
* All IDs are string.
"""
import random
from tqdm import tqdm
import os

# input
in_dir = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/graph_def'
edge_dirs = [
    os.path.join(in_dir, 'politifact', 'fake'),
    os.path.join(in_dir, 'politifact', 'real'),
    os.path.join(in_dir, 'gossipcop', 'fake'),
    os.path.join(in_dir, 'gossipcop', 'real'),
]
node_types = ['n', 'p', 'u']
edge_files = {
    ('n', 'n'): 'news-news edges.txt',
    ('n', 'p'): 'news-post edges.txt',
    ('p', 'u'): 'post-user edges.txt',
    ('u', 'u'): 'user-user edges.txt',
}
edges_to_enforce = {  # must be in the output neighbor file
    ('p', 'u'),
}

# for random walk with restart
restart_rate = 0.5
min_neigh = {
    'n': 50,
    'p': 50,
    'u': 1000,
}
num_neigh_to_record = 1200
max_steps = 10000

# for neighbor selection
max_uniq_neigh = {
    'n': 5,
    'p': 5,
    'u': 100,
}

# output
configuration_tag = 'fnn_tmp_' + \
    '_'.join([f'{k}{max_uniq_neigh[k]}' for k in node_types])
output_dir = f"rwr_results/{configuration_tag}"
print('Files output to', output_dir)

def random_walk_with_restart():
    adj_list, nei_list, involved, nodes = dict(), dict(), dict(), dict()
    for t1 in node_types:
        involved[t1] = set()  # OUT involved['p'] = {'p123', 'p456', ...}
        nodes[t1] = set()     # OUT nodes['p'] = {'p123', 'p456', ...}

    def add_adjacent(m, n):  # IN  adj_list['p123'] = ['u456', 'n789', ...]
        if m not in adj_list.keys():
            adj_list[m] = []
        adj_list[m].append(n)

    def try_add_neighbor(start_node, cur_node, num_neighs):
        t = cur_node[0]
        if len(nei_list[start_node][t]) < min_neigh[t] or \
                all([len(nei_list[start_node][s]) >= min_neigh[s] for s in node_types if s != t]):
            nei_list[start_node][t].append(cur_node)
            return num_neighs + 1
        else:
            return num_neighs

    def get_top_k_most_frequent(neighbors, k, exclude):
        counter = dict()
        for node in neighbors:
            if node not in counter:
                counter[node] = 0
            counter[node] += 1
        counter.pop(exclude, None)
        items = sorted(list(counter.items()), key=lambda x: -x[1])
        neighbors[:] = [items[i][0] for i in range(min(k, len(items)))]
    
    def enforce_edges(top_k, node, nn, k):
        for neig in adj_list[node]:
            if neig[0] == nn:
                top_k.insert(0, neig)
        top_k[:] = top_k[:k]

    def write_neighbor(node):
        for nn in node_types:
            get_top_k_most_frequent(nei_list[node][nn], max_uniq_neigh[nn], exclude=node)
            if (node[0], nn) in edges_to_enforce or (nn, node[0]) in edges_to_enforce:
                enforce_edges(nei_list[node][nn], node, nn, max_uniq_neigh[nn])

    def rwr(nodes_set):
        for start_node in tqdm(nodes_set):
            cur_node = start_node
            num_neighs, steps = 0, 0
            while num_neighs < num_neigh_to_record and steps < max_steps:
                rand_p = random.random()  # return p
                if rand_p < restart_rate:
                    cur_node = start_node
                else:
                    cur_node = random.choice(adj_list[cur_node])
                    num_neighs = try_add_neighbor(
                        start_node, cur_node, num_neighs)
                steps += 1
            write_neighbor(start_node)
            for t, nl in nei_list[start_node].items():
                involved[t] = involved[t].union(nl)

    def finalize_output():
        for node, tn in nei_list.items():
            involved[node[0]].add(node)
            for t, nl in tn.items():
                involved[t] = involved[t].union(nl)
                while len(nei_list[node][t]) < min_neigh[t]:
                    nei_list[node][t].append('<PADDING>')
    
    def compute_stats():
        stats = {t1: {t2: [] for t2 in node_types} for t1 in node_types}
        for n1, v in nei_list.items():
            for t2, x in v.items():
                stats[n1[0]][t2].append(len(x))
        for t1 in node_types:
            for t2 in node_types:
                print('stats', t1, t2, '{:.6f}'.format(
                    sum(stats[t1][t2]) / len(stats[t1][t2])))

    print("Read the graph...")
    for edge_dir in edge_dirs:
        for (main_type, neig_type), edge_f in edge_files.items():
            with open(os.path.join(edge_dir, edge_f), "r") as f:
                for l in tqdm(f.readlines(), desc=main_type+' '+neig_type):
                    l = l.strip().split()
                    if len(l) != 2:
                        break  # gossipcop real does not have user edges for now
                    add_adjacent(main_type + l[0], neig_type + l[1])
                    add_adjacent(neig_type + l[1], main_type + l[0])
                    nodes[main_type].add(main_type + l[0])
                    nodes[neig_type].add(neig_type + l[1])
    for t1 in node_types:
        for node in nodes[t1]:
            # OUT nei_list['p123']['u'] = ['u456', 'u789', ...]
            nei_list[node] = {t2: [] for t2 in node_types}

    print("Each node takes turns to be the starting node...")
    rwr(nodes['n'])
    rwr(involved['p'])
    rwr(involved['u'])

    print('Stats before padding:')
    compute_stats()
    finalize_output()

    print("Save the result...")
    for t in node_types:
        written = 0
        with open(os.path.join(output_dir, f'{t}_neighbors.txt'), 'w') as f:
            for node, type_neighs in nei_list.items():
                if node[0] == t:
                    f.write(node + ':')
                    for neig_type in node_types:
                        if len(type_neighs[neig_type]) > 0:
                            f.write(' ' + ' '.join(type_neighs[neig_type]))
                            written += len(type_neighs[neig_type])
                    f.write('\n')
        with open(os.path.join(output_dir, f'{t}_involved.txt'), "w") as f:
            f.write(' '.join(list(involved[t])) + "\n")
        print("type {}: {:10} neighbors written.".format(t, written))
        print("        {:10} nodes involved.".format(len(involved[t])))


if __name__ == "__main__":
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print("#" * 15 + configuration_tag + "#" * 15)
    random_walk_with_restart()
