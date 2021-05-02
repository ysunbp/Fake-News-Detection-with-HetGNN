"""
NOTE all IDs are string
"""
import pandas as pd
import os

in_dir = '/Users/shanglinghsu/Workspaces/fyp/buzzfeed-kaggle'
out_dir = '/Users/shanglinghsu/Workspaces/fyp/buzzfeed-kaggle'
node_files = {
    's' : 'BuzzFeedSource.txt',
}
edge_files = {
    ('s', 'n'): 'BuzzFeedSourceNews.txt',
    ('n', 'n'): 'BuzzFeedNewsNews.txt',
}

def process():
    """
    For buzzfeed, we only need to generate 'BuzzFeedSourceNews.txt' and 'BuzzFeedNewsNews.txt',
    where the latter is the string-match same-author relationship,
    because all other files are provided.
    """
    def read_csv():
        # 'id' is in the format 'Real_1-Webpage'
        df1 = pd.read_csv(open(os.path.join(in_dir, f'BuzzFeed_real_news_content.csv'), 'r'))
        df2 = pd.read_csv(open(os.path.join(in_dir, f'BuzzFeed_fake_news_content.csv'), 'r'))
        return df1.append(df2, ignore_index = True)

    with open(os.path.join(in_dir, 'BuzzFeedNews.txt')) as f:
        # ids converted from 'BuzzFeed_Real_1' to 'Real_1-Webpage'
        news_ids = {l.strip()[9:] + '-Webpage' : n for n, l in enumerate(f.readlines(), start=1)}

    def add(d, e):
        if e not in d:
            d[e] = 0
        d[e] += 1

    def write_edge(d, fn):
        with open(os.path.join(out_dir, fn), 'w') as f:
            for (s, t), v in d.items():
                f.write(f'{s}\t{t}\t{v}\n')
    
    def write_node(d, fn):
        with open(os.path.join(out_dir, fn), 'w') as f:
            l = sorted([(node, node_id) for node, node_id in d.items()], key=lambda x:x[1])
            for node, node_id in l:
                f.write(f'{node}\n')

    df = read_csv()
    sn_edges, nn_edges = dict(), dict()
    sources = {s: sid for sid, s in enumerate(df['source'].unique(), start=1)}
    same_author = {aa : set() for a in df.authors.unique() for aa in str(a).split(',')}

    for index, row in df.iterrows():
        add(sn_edges, (sources[row['source']], news_ids[row['id']]))
        for a in str(row['authors']).split(','):
            same_author[a].add(news_ids[row['id']])

    for news in same_author.values():
        for n1 in news:
            for n2 in news:
                add(nn_edges, (n1, n2))

    write_edge(sn_edges, edge_files[('s', 'n')])
    write_edge(nn_edges, edge_files[('n', 'n')])
    write_node(sources, node_files['s'])


if __name__ == '__main__':
    process()