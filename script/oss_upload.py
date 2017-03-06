# -*- coding: utf-8 -*-

import argparse
import base64
import hashlib
import os
import sys

import oss2
from oss2 import SizedFileAdapter, determine_part_size
from oss2.models import PartInfo

# 以下代码展示了文件上传的高级用法，如断点续传、分片上传等。
# 基本的文件上传如上传普通文件、追加文件，请参见object_basic.py


# 首先初始化AccessKeyId、AccessKeySecret、Endpoint等信息。
# 通过环境变量获取，或者把诸如“<你的AccessKeyId>”替换成真实的AccessKeyId等。
#
# 以杭州区域为例，Endpoint可以是：
#   http://oss-cn-hangzhou.aliyuncs.com
#   https://oss-cn-hangzhou.aliyuncs.com
# 分别以HTTP、HTTPS协议访问。
access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', 'cuRku5y77efvgqHn')
access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', '195yj5K996wrGTQsoa6f3Pwbuk1CwE')
bucket_name = os.getenv('OSS_TEST_BUCKET', 'haitajia2015log')
endpoint = os.getenv('OSS_TEST_ENDPOINT', 'http://oss-cn-hangzhou-internal.aliyuncs.com')


# 确认上面的参数都填写正确了

def main(args):

    filename = args.upload_file
    key = os.path.split(filename)[1]
    for param in (access_key_id, access_key_secret, bucket_name, endpoint):
        assert '<' not in param, '请设置参数：' + param

# 创建Bucket对象，所有Object相关的接口都可以通过Bucket对象来进行
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)

    total_size = os.path.getsize(filename)
    part_size = determine_part_size(total_size, preferred_size=100000 * 1024)

    # 初始化分片
    print('OSS initializing:')
    encode_md5 = calculate_file_md5(filename)
    print('%s s md5 is %s' % (filename, encode_md5))
    upload_id = bucket.init_multipart_upload(key).upload_id
    parts = []

    # 逐个上传分片
    print('Seperate the file batch and uploading:')
    with open(filename, 'rb') as fileobj:
        part_number = 1
        offset = 0
        while offset < total_size:
            num_to_upload = min(part_size, total_size - offset)
            result = bucket.upload_part(key, upload_id, part_number,
                                        SizedFileAdapter(fileobj, num_to_upload), progress_callback=percentage)
            parts.append(PartInfo(part_number, result.etag))
            offset += num_to_upload
            part_number += 1

    # 完成分片上传
    print('Complete the uploading task:')
    bucket.complete_multipart_upload(key, upload_id, parts)

    print('Verification the file size:')
    bucket.put_object_from_file(key, filename, headers={'Content-MD5': encode_md5})

    print ('upload complete with the file %s' % key)


def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate))
        sys.stdout.flush()

def calculate_file_md5(file_name, block_size=64 * 1024):
    """计算文件的MD5
    :param file_name: 文件名
    :param block_size: 计算MD5的数据块大小，默认64KB
    :return 文件内容的MD5值
    """
    with open(file_name, 'rb') as f:
        md5 = hashlib.md5()
        while True:
            data = f.read(block_size)
            if not data:
                break
            md5.update(data)

    return base64.b64encode(md5.digest())

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('upload_file', type=str,
        help='file for uploading to Aliyun OSS.')

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))