#!/usr/bin/env python2.7
# coding: utf-8
import os
from os.path import join, exists
import multiprocessing
import hashlib


files = '../data/dpchallenge_dataset.txt'
dp_url = 'http://www.dpchallenge.com/image.php?IMAGE_ID='
RESULT_ROOT = '~/tmp/datasets/dpc_datasets'
base_root = os.path.expanduser(RESULT_ROOT)
if not exists(base_root):
    os.mkdir(base_root)
steps = 1000

def download((urls, means)):
    """
        download from urls into folder names using wget
    """

    # download using external wget
    CMD = 'wget -c -t 1 -T 3 "%s" -O "%s"'
    for i in range(len(urls)):
        fname = hashlib.sha1(urls[i]).hexdigest() + '-' + means[i] + '.jpg'
        dst = join(base_root, fname)
        print "downloading", dst
        if exists(dst):
            print "already downloaded, skipping..."
            continue
        else:
            res = os.system(CMD % (urls[i], dst))


if __name__ == '__main__':
    urls = []
    means =[]
    with open(files, 'r') as fd:
            # strip first line
            #fd.readline()
            #names = []
            #urls = []
            #bboxes = []
        for line in fd.readlines():
            components = line.split(' ')
            assert(len(components) == 14)
            url = dp_url + components[1]
            mean = str('%.6s' % components[3].replace('.', '_'))
           # mean = mean[]
            urls.append(url)
            means.append(mean)

    task_urls = []
    task_means = []
    tasks = []
    for i in range(len(urls)):
        if i != 0 and i % steps == 0:
            tasks.append((task_urls, task_means))
            task_urls=[]
            task_means=[]

        task_urls.append(urls[i])
        task_means.append(means[i])

    tasks.append((task_urls, task_means))

    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
    pool.map(download, tasks)
    pool.close()
    pool.join()
