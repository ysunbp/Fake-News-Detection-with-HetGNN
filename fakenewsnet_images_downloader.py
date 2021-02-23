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
sources = ['politifact', 'gossipcop']
labels = ['fake', 'real']


def download_some_images(pathes: List[str], process_idx: int):
    if process_idx == 0:
        pathes = tqdm(pathes, desc='Downloading images')
    for path in pathes:
        content_path = os.path.join(path, 'news content.json')
        if not os.path.isfile(content_path):
            continue

        with open(content_path, 'r') as fin:
            news_content = json.load(fin)
            url = news_content['top_img']
        
        if url == '':
            continue

        news_id = content_path.split(os.path.sep)[-2]
        last_question_mark = url.rfind('?')
        u = url
        if last_question_mark > 0:
            u = u[:last_question_mark]
        file_ext = u.split('.')[-1]
        out_path = os.path.join(out_dir, f'{news_id}.{file_ext}')

        if os.path.isfile(out_path):  # and os.path.getsize(out_path) > 0:
            continue

        try:
            response = requests.get(url)
            if response.status_code >= 400:
                print(f'ERROR response {news_id} {response.status_code}')
            else:
                file = open(out_path, "wb")
                file.write(response.content)
                file.close()
        except Exception as e:
            print(f'ERROR unknown {news_id} {url} {repr(e)}')


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
