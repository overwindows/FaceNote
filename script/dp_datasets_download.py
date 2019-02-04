#!/usr/bin/env python2.7
# coding: utf-8
import os
from os.path import join, exists
import multiprocessing
import hashlib
import urllib
from urllib2 import HTTPError, URLError
from httplib import HTTPException
import sys

files = ['../data/photonet_dataset.txt']
dp_url = 'http://www.dpchallenge.com/image.php?IMAGE_ID='
photo_url = 'http://gallery.photo.net/photo/'
RESULT_ROOT = '~/tmp/datasets/ava_datasets'
base_root = os.path.expanduser(RESULT_ROOT)
error_path = join(base_root, 'error.log')
if not exists(base_root):
    os.mkdir(base_root)
steps = 5000

def download((urls, means)):
    """
        download from urls into folder names using wget
    """

    # download using external wget
    # CMD = 'wget -c -t 1 -T 3 "%s" -O "%s"'
    for i in range(len(urls)):
        fname = hashlib.sha1(urls[i]).hexdigest() + '-' + means[i] + '.jpg'
        dst = join(base_root, fname)
        print "downloading", dst
        if exists(dst):
            print "already downloaded, skipping..."
            continue
        else:
            try:
                urllib.urlretrieve(urls[i], dst)
            except (HTTPException, HTTPError, URLError, IOError, ValueError, IndexError, OSError) as e:
                error_message = '{}: {}'.format(urls[i], e)
                save_error_message_file(error_path, error_message)

def save_error_message_file(filename, error_message):
    print(error_message)
    with open(filename, "w") as textfile:
        textfile.write(error_message)


if __name__ == '__main__':

    for f in files:
        with open(f, 'r') as fd:
            urls = []
            means = []
            for line in fd.readlines():
                components = line.split(' ')
                # dpchallenge dataset
                if os.path.split(f)[1] == 'dpchallenge_dataset.txt':
                    url = dp_url + components[1]
                    mean = str('%.6s' % components[3].replace('.', '_'))
                else:
                    url = photo_url + components[1] + '-md.jpg'
                    mean = str('%.6s' % components[3].replace('.', '_'))
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

        try:
            # pool_size = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=2, maxtasksperchild=1)
            pool.map(download, tasks)
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            sys.exit()

