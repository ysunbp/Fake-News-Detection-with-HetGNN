import os
import torch
from tqdm import tqdm
import pandas as pd
from text_embedder import TextEmbedder
from weibo import save_embed_file

# input
in_dir = '/rwproject/kdd-db/20-rayw1/buzzfeed-kaggle'
involved_dir = f'/rwproject/kdd-db/20-rayw1/fyp_code/rwr_results/buzzfeed_n5_p5_u100'

# output
out_dir = in_dir + '/text_embeddings'
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

configs = {
    'news_titles' : {
        # Pretrained by https://medium.com/swlh/writing-buzzfeed-articles-with-fine-tuned-gpt-2-transformer-models-158fdd921788
        'model name' : 'jordan-m-young/buzz-article-gpt-2',
        'batch size' : 32,
    },
    'news_text' : {
        # Pretrained by https://medium.com/swlh/writing-buzzfeed-articles-with-fine-tuned-gpt-2-transformer-models-158fdd921788
        'model name' : 'jordan-m-young/buzz-article-gpt-2',
        'batch size' : 2,
    },
}
device = 'cuda:0'

def read_csv():
    # 'id' is in the format 'Real_1-Webpage'
    df1 = pd.read_csv(open(os.path.join(in_dir, f'BuzzFeed_real_news_content.csv'), 'r'))
    df2 = pd.read_csv(open(os.path.join(in_dir, f'BuzzFeed_fake_news_content.csv'), 'r'))
    return df1.append(df2, ignore_index = True)

def embed_text(ids, texts, max_seq_len, config, dir_name):
    def save_embeddings(ids, features):
        dir = os.path.join(out_dir, dir_name)
        if not os.path.isdir(dir):
            os.mkdir(dir)
        for i in tqdm(range(len(ids)), desc='save embed'):
            save_embed_file(os.path.join(out_dir, dir_name), ids[i], features[i])
    print('embed_text...')
    batch_size = config['batch size']
    num_batches = (len(texts) + batch_size - 1) // batch_size
    embedder = TextEmbedder(max_seq_len, config['model name'], device=device)
    embedder.tokenizer.pad_token = embedder.tokenizer.eos_token
    features = torch.zeros(len(texts), embedder.embed_dim)
    for i in tqdm(range(num_batches), desc='embed text'):
        mn = i * batch_size
        mx = min(len(texts), (i + 1) * batch_size)
        features[mn:mx] = embedder(texts[mn:mx])[:, -1, :].squeeze(1)
    save_embeddings(ids, features)

if __name__ == '__main__':
    df = read_csv()
    ids = [str(id)[:-8] for id in df['id']]
    embed_text(ids, df['title'].tolist(), max_seq_len=49, config=configs['news_titles'], dir_name = 'news_titles')
    embed_text(ids, df['text'].tolist(), max_seq_len=490, config=configs['news_text'], dir_name = 'news_text')