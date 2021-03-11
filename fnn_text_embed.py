import os
import json
import torch
from matplotlib.pyplot import text
from tqdm import tqdm
from text_embedder import TextEmbedder
from multiprocessing import Manager, Process

# input
in_dir = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/code/fakenewsnet_dataset'
ds_dirs = [
    os.path.join(in_dir, 'politifact', 'fake'),
    os.path.join(in_dir, 'politifact', 'real'),
    os.path.join(in_dir, 'gossipcop', 'fake'),
    os.path.join(in_dir, 'gossipcop', 'real'),
]
involved_dir = '/rwproject/kdd-db/20-rayw1/fyp_code/rwr_results/fnn_n5_p5_u100'  ##############

# output
out_dir = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/text_embeddings'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

configs = {
    'news' : {
        'model name' : 'mrm8488/t5-base-finetuned-summarize-news',
        'batch size' : 4,  # X: 32, 8, O: 2, 4
    },
    'tweet' : {
        'model name' : 'vinai/bertweet-base',
        'batch size' : 256,  # X: 1024, 512, O: 32, 64, 128, 256
    },
}

def embed_text(text_list, max_seq_len, config):
    batch_size = config['batch size']
    num_batches = (len(text_list) + batch_size - 1) // batch_size
    embedder = TextEmbedder(max_seq_len, config['model name'])
    outputs = torch.zeros((len(text_list), embedder.embed_dim))
    for i in tqdm(range(num_batches), desc='embed_text'):
        mn = i * batch_size
        mx = min(len(text_list), (i + 1) * batch_size)
        outputs[mn:mx] = embedder(text_list[mn:mx])[:, 0, :].squeeze(1)
    del embedder
    return outputs

def write_embed_file(fname, ids, features):
    with open(os.path.join(out_dir, fname), 'w') as f:
        for i in tqdm(range(len(ids)), desc='writing ' + fname):
            f.write('{} {}\n'.format(
                ids[i],
                ' '.join(['{:.8f}'.format(v) for v in features[i]])
            ))

def process_news():
    print('process_news')
    with open(os.path.join(involved_dir, 'n_involved.txt'), 'r') as f:
        involved_ids = set(f.readlines()[0].strip().split(' '))
    io_dict = {k : [] for k in ['ids', 'titles', 'text']}
    for ds in ds_dirs:
        for news_id in tqdm(os.listdir(ds), desc='reading ' + ds):
            content_fn = os.path.join(ds, news_id, 'news content.json')
            if 'n' + news_id not in involved_ids or not os.path.isfile(content_fn):
                continue
            with open(content_fn, 'r') as f:
                content = json.load(f)
            io_dict['ids'].append(news_id)
            io_dict['titles'].append(content['title'])
            io_dict['text'].append(content['text'])
    # title_stat = TextEmbedder.compute_seq_len_statistics(io_dict['titles'], configs['news'])
    # text_stat = TextEmbedder.compute_seq_len_statistics(io_dict['text'], configs['news'])
    io_dict['title embeddings'] = embed_text(
        io_dict['titles'],
        max_seq_len=49,
        config=configs['news'],
    )
    io_dict['text embeddings'] = embed_text(
        io_dict['text'],
        max_seq_len=490,
        config=configs['news'],
    )
    write_embed_file('news_titles.txt', io_dict['ids'], io_dict['title embeddings'])
    write_embed_file('news_text.txt', io_dict['ids'], io_dict['text embeddings'])

def process_tweets():
    print('process_tweets')
    with open(os.path.join(involved_dir, 'p_involved.txt'), 'r') as f:
        involved_ids = set(f.readlines()[0].strip().split(' '))
    io_dict = {k : [] for k in ['ids', 'text']}
    for ds in ds_dirs:
        for news_id in tqdm(os.listdir(ds), 'reading ' + ds):
            tweets_dir = os.path.join(ds, news_id, 'tweets')
            if not os.path.isdir(tweets_dir):
                continue
            for tweets_fn in os.listdir(tweets_dir):
                if 'p' + tweets_fn.split('.')[0] not in involved_ids:
                    continue
                with open(os.path.join(tweets_dir, tweets_fn), 'r') as f:
                    tweet = json.load(f)
                    io_dict['ids'].append(tweet["id_str"])
                    io_dict['text'].append(tweet['text'])
    # tweet_stat = TextEmbedder.compute_seq_len_statistics(io_dict['text'], configs['tweet'])
    io_dict['text embeddings'] = embed_text(
        io_dict['text'],
        max_seq_len=49,
        config=configs['tweet'],
    )
    write_embed_file('tweet_text.txt', io_dict['ids'], io_dict['text embeddings'])

def process_user_description():
    print('process_user_description')    
    with open(os.path.join(involved_dir, 'u_involved.txt'), 'r') as f:
        involved_uids = set(f.readlines()[0].strip().split(' '))
    i_set = set()
    for ds in ds_dirs:
        for news_id in tqdm(os.listdir(ds), 'reading ' + ds):
            tweet_dir = os.path.join(ds, news_id, 'tweets')
            if os.path.isdir(tweet_dir):
                for tweets_fn in os.listdir(tweet_dir):
                    with open(os.path.join(tweet_dir, tweets_fn), 'r') as f:
                        tweet = json.load(f)
                        if 'u' + tweet['user']["id_str"] in involved_uids:
                            i_set.add((tweet['user']["id_str"], tweet['user']['description']))
            retweet_dir = os.path.join(ds, news_id, 'retweets')
            if os.path.isdir(retweet_dir):
                for tweets_fn in os.listdir(retweet_dir):
                    with open(os.path.join(retweet_dir, tweets_fn), 'r') as f:
                        for retweet in json.load(f)['retweets']:
                            if 'u' + retweet['user']["id_str"] in involved_uids:
                                i_set.add((retweet['user']["id_str"], retweet['user']['description']))
    io_dict = {
        'ids' : [i for i, j in i_set],
        'description' : [j for i, j in i_set],
    }
    # user_stat = TextEmbedder.compute_seq_len_statistics(io_dict['description'], configs['tweet'])
    io_dict['description embeddings'] = embed_text(
        io_dict['description'],
        max_seq_len=49,
        config=configs['tweet'],
    )
    write_embed_file('tweet_text.txt', io_dict['ids'], io_dict['description embeddings'])


if __name__ == '__main__':
    process_news()
    process_tweets()
    process_user_description()