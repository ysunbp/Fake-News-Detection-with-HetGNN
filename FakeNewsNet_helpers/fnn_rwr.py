"""
* All edges are undirected.
* All IDs are string.
"""
import random
from tqdm import tqdm
import os
from multiprocessing import Process, Manager

# overall
num_process = 16

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
configuration_tag = 'fnn_test' + \
    '_'.join([f'{k}{max_uniq_neigh[k]}' for k in node_types])  ##########################################
output_dir = f"rwr_results/{configuration_tag}"

def update_involved_recursively(nei_list, rank):
    if len(nei_list) == 0:
        return {t : set() for t in node_types}
    if rank == 0:
        pbar = tqdm(total=len(nei_list) * 2 - 1, desc='update_involved')
    def recursion(nei_list, nodes):
        # speedup intuition: ∑_{i=1...2^3} log(i) < ∑_{i=0...k} 2^i * log(2^{k-i})
        # nei_list is pass-by-ref, but nodes is copied in every level
        involved = dict()
        if len(nodes) == 1:
            for t, nl in nei_list[nodes[0]].items():
                involved[t] = set(nl)
        else:
            m = len(nodes) // 2
            l = recursion(nei_list, nodes[:m])
            r = recursion(nei_list, nodes[m:])
            for t in node_types:
                involved[t] = l[t].union(r[t])
        if rank == 0:
            pbar.update(1)
        return involved
    if rank == 0:
        pbar.close()
    return recursion(nei_list, list(nei_list.keys()))

def rwr_worker(adj_list, nodes_list, desc, rank, nei_list_subsets, involved_subsets):
    # Each process uses a complete adj_list to construct an incomplete nei_list.
    nei_list = {node : {t : [] for t in node_types} for node in nodes_list}  # OUT nei_list['p123']['u'] = ['u456', 'u789', ...]

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
            if neig[0] == nn and neig not in top_k:
                top_k.insert(0, neig)
        top_k[:] = top_k[:k]

    def write_neighbor(node):
        for nn in node_types:
            get_top_k_most_frequent(nei_list[node][nn], max_uniq_neigh[nn], exclude=node)
            if (node[0], nn) in edges_to_enforce or (nn, node[0]) in edges_to_enforce:
                enforce_edges(nei_list[node][nn], node, nn, max_uniq_neigh[nn])
    
    it = tqdm(nodes_list, desc=desc + ' proc 0') if rank == 0 else nodes_list
    for start_node in it:
        cur_node = start_node
        num_neighs, steps = 0, 0
        while num_neighs < num_neigh_to_record and steps < max_steps:
            rand_p = random.random()  # return p
            if rand_p < restart_rate:
                cur_node = start_node
            else:
                cur_node = random.choice(adj_list[cur_node])
                num_neighs = try_add_neighbor(start_node, cur_node, num_neighs)
            steps += 1
        write_neighbor(start_node)
    
    involved = update_involved_recursively(nei_list, rank)
    nei_list_subsets[rank] = nei_list
    involved_subsets[rank] = involved

def update_involved_worker(nei_list, rank, involved_subsets):
    # each process has its own subset of nei_list
    involved = update_involved_recursively(nei_list, rank)
    involved_subsets[rank] = involved

def save_result_worker(nei_list, involved, t, return_dict):
    max_val = max(list(max_uniq_neigh.values()))
    mode_type = [k for k, v in max_uniq_neigh.items() if v == max_val][0]

    written = 0
    with open(os.path.join(output_dir, f'{t}_neighbors.txt'), 'w') as f:
        it = tqdm(nei_list.items(), desc=f'write {t} neigh') if t == mode_type else nei_list.items()
        for node, type_neighs in it:
            if node[0] == t:
                f.write(node + ':')
                for neig_type in node_types:
                    if len(type_neighs[neig_type]) > 0:
                        f.write(' ' + ' '.join(type_neighs[neig_type]))
                        if len(type_neighs[neig_type]) < min_neigh[neig_type]:
                            f.write(' <PADDING>' * (
                                min_neigh[neig_type] - len(type_neighs[neig_type])
                            ))
                        written += len(type_neighs[neig_type])
                f.write('\n')
    with open(os.path.join(output_dir, f'{t}_involved.txt'), "w") as f:
        f.write(' '.join(list(involved[t])) + "\n")
    ret_str = "type {}: {:10} neighbors written.\n".format(t, written) + \
              "        {:10} nodes involved.\n".format(len(involved[t]))
    return_dict[t] = ret_str

def random_walk_with_restart():
    adj_list, nei_list, involved, nodes = dict(), dict(), dict(), dict()
    nodes = {t : set() for t in node_types}     # IN  nodes['p'] = {'p123', 'p456', ...}
    involved = {t : set() for t in node_types}  # OUT involved['p'] = {'p123', 'p456', ...}

    def add_adjacent(m, n):  # IN  adj_list['p123'] = ['u456', 'n789', ...]
        if m not in adj_list.keys():
            adj_list[m] = []
        adj_list[m].append(n)

    def rwr(nodes_set, desc):
        nodes_list = list(nodes_set)
        job_sz = (len(nodes_list) + num_process - 1) // num_process
        jobs = []
        manager = Manager()
        nei_list_subsets, involved_subsets = manager.dict(), manager.dict()
        for i in range(num_process):
            mn, mx = i * job_sz, min((i + 1) * job_sz, len(nodes_list))
            p = Process(target=rwr_worker, args=(adj_list, nodes_list[mn:mx], desc, i, nei_list_subsets, involved_subsets))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        for nei_list_subset in nei_list_subsets.values():  # TODO: speed up this by recursion
            nei_list.update(nei_list_subset)
        for involved_subset in involved_subsets.values():  # TODO: speed up this by recursion
            for t in node_types:
                involved[t] = involved[t].union(involved_subset[t])

    def update_involved():
        keys = list(nei_list.keys())
        job_sz = (len(keys) + num_process - 1) // num_process
        jobs = []
        manager = Manager()
        involved_subsets = manager.dict()
        for i in range(num_process):
            mn, mx = i * job_sz, min((i + 1) * job_sz, len(keys))
            nei_list_subset = {keys[i] : nei_list[keys[i]] for i in range(mn, mx)}
            p = Process(target=update_involved_worker, args=(nei_list_subset, i, involved_subsets))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        for involved_subset in involved_subsets.values():  # TODO: speed up this by recursion
            for t in node_types:
                involved[t] = involved[t].union(involved_subset[t])
    
    def compute_stats():
        stats = {t1: {t2: [] for t2 in node_types} for t1 in node_types}
        for n1, v in tqdm(nei_list.items(), desc='compute_stats'):
            for t2, x in v.items():
                stats[n1[0]][t2].append(len(x))
        for t1 in node_types:
            for t2 in node_types:
                print('stats', t1, t2, '{:.6f}'.format(
                    sum(stats[t1][t2]) / len(stats[t1][t2])))
    
    def save_result():
        jobs = []
        manager = Manager()
        return_dict = manager.dict()
        for t in node_types:
            p = Process(target=save_result_worker, args=(nei_list, involved, t, return_dict))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        for ret_str in return_dict.values():
            print(ret_str)

    print("Read the graph...")
    for edge_dir in edge_dirs:
        print("Reading", edge_dir)
        for (main_type, neig_type), edge_f in edge_files.items():
            with open(os.path.join(edge_dir, edge_f), "r") as f:
                for l in tqdm(f.readlines(), desc='read ' + main_type+' '+neig_type):  ########################################
                    l = l.strip().split()
                    if len(l) != 2:
                        break  # gossipcop real does not have user edges for now
                    add_adjacent(main_type + l[0], neig_type + l[1])
                    add_adjacent(neig_type + l[1], main_type + l[0])
                    nodes[main_type].add(main_type + l[0])
                    nodes[neig_type].add(neig_type + l[1])
    # ############################################################################### toy input
    # for (n1, n2) in {('n1','n2'),('n1','p1'),('n2','p2'),('p1','u1'),('p1','u2'),('p2','u2'),('u1','u2')}:
    #     add_adjacent(n1, n2)
    #     add_adjacent(n2, n1)
    #     nodes[n1[0]].add(n1)
    #     nodes[n2[0]].add(n2)  
    # ###############################################################################

    print("Each node takes turns to be the starting node...")
    rwr(nodes['n'], 'news rwr')
    rwr(involved['p'], 'post rwr')
    rwr(involved['u'], 'user rwr')

    print('Stats before padding:')
    compute_stats()
    update_involved()

    print("Save the result...")
    save_result()


if __name__ == "__main__":
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print("\n" + " - " * 10 + configuration_tag + " - " * 10 + "\n")
    print('Files output to', output_dir)
    random_walk_with_restart()
