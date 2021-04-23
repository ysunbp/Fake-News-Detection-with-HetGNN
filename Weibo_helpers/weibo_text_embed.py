# See Readme.md for the docs
import os
import time
from multiprocessing import Process

import torch
from tqdm import tqdm


############################################################
#                   Embed text - w2v                       #
############################################################    

def save_embed_file(dir, tid, feature):
    f = open('{}/{}.txt'.format(dir, tid), 'w')
    f.writelines(' '.join(['{:.8f}'.format(v) for v in feature]) + '\n')
    f.close()

def embed_text_list_w2v_save(tokenizer, w2v, tid_list, text_list, output_dir, process_idx = 0):
    """Embed par with pretrained w2v"""
    
    print("Process {} says hi!".format(process_idx))
    batch_size = 32
    n_sent = len(text_list)
    hidden_size = 300 # const, not var. set by pretrained w2v.

    def embed(tokens):
        wv_sum = torch.zeros(hidden_size)  # no valid words => zeros
        ctr = 0
        for token in tokens:
            vec = w2v.get(token)
            if vec != None:
                wv_sum += torch.FloatTensor(vec)
                ctr += 1
        if ctr > 0:  # to avoid nan caused by division by 0
            wv_sum /= ctr
        return wv_sum

    # Batchified tokenization
    n = n_sent // batch_size
    it = tqdm(range(n), desc='Process 0 progress') if process_idx == 0 else range(n)
    for i in it:    
        tids = tid_list[i * batch_size : (i + 1) * batch_size]
        sents = tokenizer(text_list[i * batch_size : (i + 1) * batch_size])
        for tid, sent in zip(tids, sents['input_ids']):
            tokens = tokenizer.convert_ids_to_tokens(sent)
            feature = embed(tokens)
            save_embed_file(output_dir, tid, feature)
            
    # Last batch
    if len(text_list) % batch_size != 0:
        tids = tid_list[n * batch_size:]
        sents  = tokenizer(text_list[n * batch_size:])
        for tid, sent in zip(tids, sents['input_ids']):
            tokens = tokenizer.convert_ids_to_tokens(sent)
            feature = embed(tokens)
            save_embed_file(output_dir, tid, feature)
    
    print("Process {} says bye!".format(process_idx))

def multiprocess_embed_w2v(num_processes, tids, texts, w2v_path, output_dir):
    from transformers import XLMRobertaTokenizer
    
    def load_weibo_w2v():
        """Load pretrained Weibo w2v into a dict."""
        w2v = dict()
        with open(w2v_path) as w2v_file:
            lines = w2v_file.readlines()
            info, vecs = lines[0], lines[1:]
            
            info = info.strip().split()
            vocab_size, embed_dim = int(info[0]), int(info[1])

            for vec in vecs:
                vec = vec.strip().split()
                w2v[vec[0]] = [float(val) for val in vec[1:]]
        return w2v

    print("multiprocess_embed_w2v starts.")
    start = time.time()

    # XLMRobertaTokenizer is based on SentencePiece (https://github.com/google/sentencepiece)
    # If it doesn't work, switch to SentencePiece
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')  # resume_download=True if needed    
    w2v = load_weibo_w2v()

    n = (len(tids) + num_processes - 1) // num_processes  # The number of units in each process
    processes = []
    for i in range(num_processes):
        end = min((i+1) * n, len(tids))
        processes.append(
            Process(target=embed_text_list_w2v_save, args=(
                tokenizer, 
                w2v,
                tids[i * n : end],
                texts[i * n : end],
                output_dir, 
                i))
        )
        processes[-1].start()
    for p in processes:
        p.join()
    
    end = time.time()
    print("multiprocess_embed_w2v ends. Elapse:", end - start)

############################################################
#                Embed text - Transformer                  #
############################################################    
def save_embed_worker(id_list, feature_list, output_dir, process_idx = 0):
    print("Process {} says hi!".format(process_idx))
    it = tqdm(zip(id_list, feature_list), desc='save embed') if process_idx == 0 else zip(id_list, feature_list)
    for tid, feature in it:
        save_embed_file(output_dir, tid, feature)
    print("Process {} says bye!".format(process_idx))

def multiprocess_embed_transformer(num_processes, tids, texts, output_dir, transformer_path):
    """
    Use a transformer to embed some strings and save them in dir

    Requirements: transformers v3.3.0, pytorch v1.6.0
        Note: 
        1. The transformers version in conda isn't new enough, 
           so if you're using conda:
            $ deactivate conda
            $ pip3 install --user transformers
        2. When you run this function for the first time, it
           downloads the model, which takes 30+ mins over wifi.
    Parameters: 
        num_processes: int, # processes used to save embed files
        tids: list(str), the list of the id of the text
        texts: list(str), the list of text to embed
        output_dir: a dir path to save the embeddings
        transformer_path: a tranformer name or a path to the finetuned transformer
    Usage:
        text_1 = "公安内部资料显示，她的户口是由山东青岛迁出，迁往地是日本的大阪。评：古人云“苟富贵，无相忘”，现在倒好，就连张海迪这样真正是靠全党、全国人民无数双手捧起来、曾经是一代人心中楷模和英雄偶像的残疾人，也在当高官、享厚禄后，拍屁股外逃了！"
        text_2 = "#来论#【救的不只是一个日本人】中国海军在也门撤侨行动中帮助一名日本游客乘中国军舰脱离险地。中国军人该不该救那个日本国民？当然应该。漠视一个无辜的、面临死亡威胁的生命，会让我们堕落到幽暗人性的深渊。我们需要告别被害者心态，而应慢慢具备正常大国国民的平和心态。http://t.cn/RAidbJu"
        
        id_list = ['0', '1']
        text_list = [text_1, text_2]
        multiprocess_embed_transformer(8, id_list, text_list, 'data/weibo/posts', 'xlm-roberta-base')
    """
    print("multiprocess_embed_transformer starts.")    
    start = time.time()

    # Embed with GPU; single-process; batchified
    from text_embedder import TextEmbedder
    batch_size = 32
    embedder = TextEmbedder(512, 'xlm-roberta-base', transformer_path)  # 512: max num tokens for XLM-R
    features = torch.zeros((len(tids), embedder.embed_dim))
    n = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(n), desc='embed'):
        mn, mx = i * batch_size, min((i + 1) * batch_size, len(texts))
        outputs = embedder(texts[mn:mx])  # (batch_size, seq_len, embed_dim)
        features[mn:mx] = outputs['last_hidden_state'][:, 0, :].squeeze(1)  # (batch_size, embed_dim)

    # Save model with CPU, multi-process
    n = (len(tids) + num_processes - 1) // num_processes  # The number of units in each process
    processes = []
    for i in range(num_processes):
        end = min((i+1) * n, len(tids))
        processes.append(
            Process(target=save_embed_worker, args=(
                tids[i * n : end], 
                features[i * n : end], 
                output_dir,
                i))
        )
        processes[-1].start()
    for p in processes:
        p.join()
    
    end = time.time()
    print("multiprocess_embed_transformer ends. Elapse:", end - start)

############################################################
#                   Clustering & split                     #
############################################################
def one_pass_clustering(vectors, threshold):
    """
    Parameters:
        vectors: torch tensor, shape (# instances, embed_dim),
            the obejcts for clustering
        threshold: float, how different should the cluster and the new vector be so a new cluster is created for the new vector?
    Returns:
        num_clusters: int, the number of clusters created
        cluster_idx: an array of the cluster index each vector 
            belongs to

    Usage:
        import torch
        from weibo import one_pass_clustering
        t = torch.tensor([[0,0,0],[2,2,2],[1,1,1]], 
            dtype=torch.float32)
        num_clusters, cluster_idx = one_pass_clustering(t, 2)
    """
    clusters = []
    cluster_idx = []
    for vec in vectors:
        assigned = False
        for i in range(len(clusters)):
            if torch.norm(clusters[i][0] - vec) <= threshold:
                n = len(clusters[i][1])
                clusters[i][0] = \
                    (clusters[i][0] * n + vec) / (n + 1)
                clusters[i][1].append(vec)
                cluster_idx.append(i)
                assigned = True
                break
        if not assigned:
            cluster_idx.append(len(clusters))
            clusters.append([vec, [vec]])  # mean, elements
    
    num_clusters = len(clusters)
    return num_clusters, cluster_idx

def split_by_clusters(num_clusters, news_ids, cluster_idx):
    """
    "Split the whole datasets into the training, validation, 
    testing sets in a 7:1:2 ratio, and ensure that they do 
    not contain any common event."

    The unit I split is cluster rather than piece of news.
    
    Requirements: numpy
    Returns:
        ids_split: dict(str -> str), 
            news_id -> one of {"train", "valid", "test"}
        train_ids: list(str)
        valid_ids: list(str)
        test_ids: list(str)

    Usage:
        num_clusters = 10
        news_ids= [str(i) for i in range(30)]
        cluster_idx = [i for j in range(3) for i in range(10)]

        ids_split, splits, train_ids, valid_ids, test_ids = split_by_clusters(num_clusters, news_ids, cluster_idx)
    """
    import numpy as np
    arr = np.arange(num_clusters)
    arr_rand = np.random.permutation(arr)

    train_end = num_clusters // 10 * 7
    valid_end = num_clusters // 10 * (7 + 1)

    splits, train_ids, valid_ids, test_ids = [], [], [], []
    ids_split = dict()
    for nid in range(len(cluster_idx)):
        cidx = cluster_idx[nid]
        if arr_rand[cidx] < train_end:
            train_ids.append(news_ids[nid])
            splits.append("train")
            ids_split[news_ids[nid]] = "train"
        elif arr_rand[cidx] < valid_end:
            valid_ids.append(news_ids[nid])
            splits.append("valid")
            ids_split[news_ids[nid]] = "valid"
        else:
            test_ids.append(news_ids[nid])
            splits.append("test")
            ids_split[news_ids[nid]] = "test"

    return ids_split, splits, train_ids, valid_ids, test_ids

def save_split(ids_split, splits, train_ids, valid_ids, test_ids, 
        save_dir):
    import json
    with open(os.path.join(save_dir, "ids_split.json"), 'w') as outfile:
        json.dump(ids_split, outfile)
    
    with open(os.path.join(save_dir, "splits.txt"), 'w') as outfile:
        outfile.writelines('\n'.join(splits) + '\n')
    
    sets = ["train", "valid", "test"]
    ids = [train_ids, valid_ids, test_ids]
    for i in range(3):
        with open(os.path.join(save_dir, sets[i] + "_ids.txt"), 'w') as outfile:
            outfile.write(' '.join(ids[i]) + '\n')


############################################################
#                    Load Weibo dataset                    #
############################################################

def get_weibo_text(weibo_dir, users_involved_path, small_subset=-1):
    """
    * News text := 1st tweet text of each piece of news.
    * Tweet text := other tweet text
    * User text := user description text
    * It takes 2 mins on server to load all news text.

    Returns:
        news_ids: list(str), a list of news id
        news_text: list(str), a list of news text
    Usage:
        news_ids, news_text, tweet_ids, tweet_text, user_ids, \
            user_text = get_weibo_text(weibo_dir)
    """
    import json

    def read_users_involved():
        with open(users_involved_path, 'r') as fin:
            line = fin.readlines()[0]
        ids = line.strip().split()
        ids = set([int(t[1:]) for t in ids])
        return ids
        

    news_ids, news_text, tweet_ids, tweet_text, \
        user_ids, user_text = [], [], [], [], [], []
    user_id_set = set()
    users_involved = read_users_involved()
    
    def check_add_user(tweet):
        if tweet['uid'] in users_involved and tweet['uid'] not in user_id_set:
            user_id_set.add(tweet['uid'])
            user_ids.append(tweet['uid'])
            user_text.append(tweet['user_description'])

    ctr = 0
    for root, dirs, files in os.walk(weibo_dir, topdown=False):
        for i in tqdm(range(len(files)), desc="Load Weibo"):
            name = files[i]
            # To load only a small subset for debugging
            if small_subset > 0 and ctr >= small_subset:
                break
            ctr += 1
            with open(os.path.join(root, name), 'r') as json_file:
                tweet_list = json.load(json_file)
                # original weibo post
                news_ids.append(tweet_list[0]['id'])
                news_text.append(tweet_list[0]['text'])
                check_add_user(tweet_list[0])
                # forwarded weibo posts
                for tweet in tweet_list[1:]:
                    tweet_ids.append(tweet['id'])
                    tweet_text.append(tweet['text'])
                    check_add_user(tweet)
    return news_ids, news_text, tweet_ids, tweet_text, user_ids, user_text
    

############################################################
#      Load text -> embed -> cluster -> split -> save      #
############################################################
if __name__ == "__main__":
    weibo_dir = '/rwproject/kdd-db/20-rayw1/rumdect/weibo_json'
    # users_involved_path = "users_involved.txt"
    users_involved_path = "/rwproject/kdd-db/20-rayw1/fyp_code/random_walk_gcn/5p100u/new_users.txt"
    # w2v_path = "word2vec/sgns.weibo.bigram-char"
    transformer_path = '/rwproject/kdd-db/20-rayw1/language_models/xlm-roberta-base'
    output_dir = "../data/weibo/xlm-roberta-base-5p100u/"
    # output_dir = "data/weibo/"
    output_dirs = [
        output_dir, 
        output_dir + 'posts/', 
        output_dir + 'users/'
    ]

    num_processes = 8
    # For w2v / tweets / Raymond's server, the estimated time:
    #   16    9h
    #   32    6h (*)
    #   64    6h
    # For transformer / users involved / Raymond's server, the estimated time:
    #   8       1h (*)
    #   16      1.5h

    for dir in output_dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    print("Load all weibo data")  # 38 sec on server
    news_ids, news_text, tweet_ids, tweet_text, user_ids, \
        user_text = get_weibo_text(weibo_dir, users_involved_path)

    ############################################################
    #            Encode stuffs with a Transformer.             #
    ############################################################
    # print("Embed news Transformer: {} processes".format(num_processes))
    # multiprocess_embed_transformer(num_processes, news_ids, news_text, output_dirs[1])
    
    print("Embed users Transformer: {} processes".format(num_processes))
    multiprocess_embed_transformer(num_processes, user_ids, user_text, output_dirs[2], transformer_path)


    ############################################################
    #                 Encode stuffs with w2v.                  #
    ############################################################
    # print("Embed tweets w2v: {} processes".format(num_processes))
    # multiprocess_embed_w2v(num_processes, tweet_ids, tweet_text, w2v_path, output_dirs[1])
    
    # print("Embed users w2v: {} processes".format(num_processes))
    # multiprocess_embed_w2v(num_processes, user_ids, user_text, w2v_path, output_dirs[2])


    ############################################################
    #         The following code is not used for now.          #
    ############################################################
    # threshold = 0.2  # I tried a few diff vals. This works.
    
    # print("Cluster")  # 160 sec on server
    # num_clusters, cluster_idx = \
    #     one_pass_clustering(news_vecs, threshold)
    # print(f"  {num_clusters} clusters")  # 4394 for weibo-4664
    
    # print("Split")
    # ids_split, splits, train_ids, valid_ids, test_ids = \
    #     split_by_clusters(num_clusters, news_ids, cluster_idx)
    # save_split(ids_split, splits, train_ids, valid_ids, test_ids, 
    #     output_dir) 