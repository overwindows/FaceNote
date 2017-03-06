#encoding:utf-8
'''Clean Youtube face datasets as same as facesrab and vgg_face_datasets'''

import os, sys
import shutil
from shutil import copy2
import math


srcDir = './youtubeface_datasets/'
dstDir = './youtube_facesets/'
batch_size = 1000


def get_path(srcfile, dstdir):
    src_file = os.path.expanduser(srcfile)
    src_file_lists=[]
    for root, dirs, files in os.walk(src_file, topdown=False):
        for name in files:
            src_file_name = os.path.join(root, name)
            src_file_lists.append(src_file_name)
        #for name in dirs:
         #   print(os.path.join(root, name))

    dst_dir_lists = []
    listdirs = os.listdir(src_file)
    for f in listdirs:
        path = os.path.join(srcfile, f)
        if not os.path.isfile(os.path.expanduser(path)):
            dst_dir = os.path.join(os.path.expanduser(dstdir), f)
            dst_dir_lists.append(dst_dir)

    copies = []
    for i in range (len(src_file_lists)):
        for j in range(len(dst_dir_lists)):
            if os.path.split(dst_dir_lists[j])[1] in src_file_lists[i]:
                copies.append((src_file_lists[i], dst_dir_lists[j]))
            # print(copies)
    return copies


def copy_file(src_file, dst_dir):
    '''声明函数 copy_file( 要复制的文件，目标目录，复制符号连接内容到新目录，没有要忽略文件)'''

    '''复制一个文件中所以的东西到另一个目录中'''
    if os.path.isfile(src_file):
        if os.path.isdir(dst_dir):  # dst目标存在时，就pass,不存在就创建
            pass
        else:
            os.makedirs(dst_dir)

        errors = []  # 声明 errors列
        print src_file
        srcname = src_file
        filename = os.path.basename(src_file)
        dstname = os.path.join(dst_dir, filename)  # 将路径名（dst）添加到文名（name）之前然后赋值给 dstcname
        from shutil import Error
        try:  # 尝试
            if os.path.isfile(srcname):  # 如果srcname文件是存在
                copy2(srcname, dstname)
                print srcname, dstname, 'success'
            elif os.path.isfile(dstname):  # 目标文件存在，删除后，再复制新的
                os.remove(dstname)
                print 'remove %s' % dstname
                copy2(srcname, dstname)

        except (IOError, os.error), why:  # 除（IOError［与文件有关的异常］，操作系统异常）外，返回原因
            errors.append((srcname, dstname, str(why)))  # 向errors列里添加，（要复制的目录，目标目录，错误原因）
        # catch the Error from the recursive jiecptree so that we can  从递归复制中捕捉这个错误，以便于我们能继续复制其他文件
        # continue with other files
        except Error, err:  # 除错误外，返回错误：
            errors.extend(err.args[0])  # 扩展 errors 列，添加（err.args[0] 元素）


if __name__ == '__main__':

    copy_command = get_path(srcDir, dstDir)
    nrof_command = len(copy_command)
    nrof_batches = int(math.ceil(1.0*nrof_command / batch_size))
    nrof_print = 0
    for i in range(nrof_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_command)
        for j in range(start_index, end_index):
            copy_file(copy_command[j][0], copy_command[j][1])
            nrof_print += 1
        print ("this batch copied of files: %s files" % nrof_print)
    print ("Total copied of files: %s files" % nrof_print)

