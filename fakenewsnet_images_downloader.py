import os
import json
from multiprocessing import Process
from os.path import isfile
from typing import List
from random import shuffle
from tqdm import tqdm
import requests

num_processes = 8
in_dir = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/code/fakenewsnet_dataset'
out_dir = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/top_images'
err_res_path = '/rwproject/kdd-db/20-rayw1/FakeNewsNet/img_err_res_news_id.log'
sources = ['politifact', 'gossipcop']
labels = ['fake', 'real']

img_file_exts = ["JPG", "PNG", "GIF", "WEBP", "TIFF", "PSD", "RAW", "BMP", "HEIF", "JPEG", "SVG", "AI", "EPS", "PDF"]

def download_some_images(pathes: List[str], process_idx: int):
    
    new_err_res_ids = set()
    with open(err_res_path, 'r') as fin:
        err_res_ids = set([e.strip() for e in fin.readlines()])

    if process_idx == 0:
        pathes = tqdm(pathes, desc='Downloading images')
    for pi, path in enumerate(pathes):
        if pi % 50 == (process_idx * 7) % 50:
            with open(err_res_path, 'r') as fin:
                old_ids = set([e.strip() for e in fin.readlines()])
            new_err_res_ids = new_err_res_ids.union(old_ids)
            with open(err_res_path, 'w') as fout:
                fout.write('\n'.join(new_err_res_ids) + '\n')
            new_err_res_ids = set()

        content_path = os.path.join(path, 'news content.json')
        news_id = content_path.split(os.path.sep)[-2]
        if not os.path.isfile(content_path):
            print(f'DEBUG no_content {news_id}')
            new_err_res_ids.add(news_id)
            continue

        if news_id in err_res_ids:
            print(f'DEBUG in_err_res_ids {news_id}')
            new_err_res_ids.add(news_id)
            continue

        with open(content_path, 'r') as fin:
            news_content = json.load(fin)
            url = news_content['top_img']
        
        if url == '':
            print(f'DEBUG empty_url {news_id}')
            new_err_res_ids.add(news_id)
            continue

        file_ext = 'UNK_EXT'
        for ext in img_file_exts:
            ext_idx_upper = url.rfind(ext)
            ext_idx_lower = url.rfind(ext.lower())
            if ext_idx_upper != -1 or ext_idx_lower != -1:
                file_ext = ext
                break
        out_path = os.path.join(out_dir, f'{news_id}.{file_ext}')

        if os.path.isfile(out_path):  # and os.path.getsize(out_path) > 0:
            print(f'DEBUG processed {news_id}')
            # new_err_res_ids.add(news_id)
            continue

        try:
            response = requests.get(url)
            if response.status_code >= 400:
                print(f'DEBUG response {news_id} {response.status_code}')
                new_err_res_ids.add(news_id)
            else:
                file = open(out_path, "wb")
                file.write(response.content)
                file.close()
        except Exception as e:
            print(f'ERROR unknown {news_id} {url} {repr(e)}')
            new_err_res_ids.add(news_id)


def download_images(paths: List[str]):
    processes = []
    num_files_per_processes = \
        (len(paths) + num_processes - 1) // num_processes
    for i in range(num_processes):
        paths_subset = paths[i *
                             num_files_per_processes: min((i +
                                                           1) *
                                                          num_files_per_processes, len(paths))]
        processes.append(Process(target=download_some_images,
                                 args=(paths_subset, i)))
        processes[-1].start()
    for p in processes:
        p.join()
    print("Finished downloading images")


if __name__ == '__main__':
    paths = []
    for s in sources:
        for l in labels:
            prefix = os.path.join(in_dir, s, l)
            pathes = os.listdir(prefix)
            paths.extend([os.path.join(prefix, path) for path in pathes])

    shuffle(paths)
    download_images(paths)
