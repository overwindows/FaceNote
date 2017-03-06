# -*- coding: utf-8 -*-

import copy
import os
import socket
import sys
import threading
import time
import urllib

import numpy as np

socket.setdefaulttimeout(15.0)

reload(sys)
sys.setdefaultencoding('utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'out')
FILE_DIR = os.path.join(BASE_DIR, 'in')
LOG_DIR = os.path.join(BASE_DIR, 'failed.log')

LOCK = threading.Lock()
FILE_LOCK = threading.Lock()
SIZE_LOCK = threading.Lock()
OUTPUT_LOCK = threading.Lock()
FINISH_LOCK = threading.Lock()

SUCCESS_COUNT = [0]
TOTAL_AMOUNT = 0
THREAD_AMOUNT = 4
COST_TIME = [0]
SUCCESS_FILE_SIZE = [0]
FINISH_THREAD_AMOUNT = [0]
DEBUG = True

# 给出待下载的文件名称，将该名称对应的文件内所有的图片下载到IMG_DIR下该名称对应的目录下
class BatchDownloader(threading.Thread):
  # 初始化时将待下载的文件名称传给downloader
  def __init__(self, nm, fl):
    super(BatchDownloader, self).__init__()
    self.name = nm
    self.fileList = copy.deepcopy(fl)

  # 将url指向的文件下载到path中
  def download(self, path, url):
    try:
      urllib.urlretrieve(url, path, process)
      return True, None
    except Exception, e:
      return False, e

  # 以此读取负责的文件中的每一行，并下载该行对应的图片
  # 如果下载失败，则将失败的url写进日志文件中
  def run(self):
    for fn in self.fileList:
      fp = os.path.join(FILE_DIR, fn + ".txt")
      _file = open(fp, 'r')
      line = _file.readline()
      lineNum = 0
      while line:
        lineNum += 1
        # logger('DEBUG', '%s readling the %dth line' % (self.name, lineNum))
        arr = line.split()
        fname = arr[0] + ".jpg"
        outPath = os.path.join(os.path.join(IMG_DIR, fn))
        dst = os.path.join(outPath, fname)
        # bbox = np.rint(np.array(map(float, arr[2:6])))
        res = [True]
        if not os.path.exists(dst):
          url = arr[1]
          res = self.download(dst, url)
        # 如果下载成功，则SUCCESS_COUNT自增1
        if LOCK.acquire():
          SUCCESS_COUNT[0] += 1
          LOCK.release()
        #face_name = os.path.split(fn)[1] + '_' + arr[0]
        #face_path = os.path.join(outPath, face_name)
        if not res[0]:
          if FILE_LOCK.acquire():
            f = open(LOG_DIR, 'a')
            f.write('%s\t%s\t%s\t%s\n' % (fn, arr[0], url, res[1]))
            f.close()
            FILE_LOCK.release()
        # get face
        #img = cv2.imread(dst)
        #if img is None:
          # no image data
        #  os.remove(dst)
        #if img.ndim == 2:
         #   img = to_rgb(img)
        #if img.ndim != 3:
         #   raise ValueError('Wrong number of image dimensions')
        #hist = np.histogram(img, 255, density=True)
        #if hist[0][0] > 0.9 and hist[0][254] > 0.9:
         #   raise ValueError('Image is mainly black or white')
        #else:
            # Crop image according to dataset descriptor

         #   face_cropped = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            # Scale to 256x256
          #  image_size = 192
           # face = cv2.resize(face_cropped, (image_size, image_size))
            # Save image as .png
            #cv2.imwrite(face_path, face)
            #with open(os.path.join(face_directory, '_bboxes.txt'), 'a') as fd:
             #   fd.write('%s %i %i %i %i\n' % (fname, bbox[0], bbox[1], bbox[2], bbox[3]))
                # no image data
            #os.remove(dst)
        line = _file.readline()
      _file.close()
    # logger('DEBUG', self.name + u' 完成')
    if FINISH_LOCK.acquire():
      FINISH_THREAD_AMOUNT[0] += 1
      FINISH_LOCK.release()

def logger(tp, msg):
  if OUTPUT_LOCK.acquire():
    if not DEBUG and tp.upper() == 'DEBUG':
      return
    print u'[%s]\t%s' % (tp, msg)
    OUTPUT_LOCK.release()

# 用于显示已下载文件的平均速度
def downloadSpeed():
  unit = 'B/s'
  speed = float(SUCCESS_FILE_SIZE[0]) / COST_TIME[0]
  if speed < 1024:
    return str(speed) + " " + unit
  speed /= 1024
  unit = "KB/s"
  if speed < 1024:
    return '%.2f %s' % (speed, unit)
  speed /= 1024
  unit = "MB/s"
  return '%.2f %s' % (speed, unit)

# 显示单个下载过程的进度
def process(a, b, c):
  # per = 100 * a * b / c
  # if per > 100:
  #   per = 100
  # print '%.1f%%' % per
  if SIZE_LOCK.acquire():
    SUCCESS_FILE_SIZE[0] += (a * b)
    SIZE_LOCK.release()

# RGB化
def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

# 显示整个下载进度
def showProcess():
  COST_TIME[0] += 1
  per = '%.2f' % (100 * float(SUCCESS_COUNT[0]) / TOTAL_AMOUNT)
  sys.stdout.write('\r[INFO]\t' + str(SUCCESS_COUNT[0]) + '/' + str(TOTAL_AMOUNT) + ' complete ' + str(per) + '%, average spped: ' + downloadSpeed() + ' Total cost: ' + prettyTime(COST_TIME[0]))
  sys.stdout.flush()
  if FINISH_THREAD_AMOUNT[0] < THREAD_AMOUNT:
    global timer
    timer = threading.Timer(1.0, showProcess)
    timer.setDaemon(True)
    timer.start()
timer = threading.Timer(1.0, showProcess)

# 格式化时间
def prettyTime(second):
  hour = 0
  minute = 0
  if second < 60:
    return str(second) + "secs"
  minute = int(second / 60)
  second = second % 60
  if minute < 60:
    return str(minute) + "mins" + str(second) + u"secs"
  hour = int(minute / 60)
  minute = minute % 60
  return str(hour) + "hours" + str(minute) + "mins" + str(second) + "secs"

# 获取输入参数
def getParams(args):
  params = ' '.join(args).split('-')
  ret = {}
  for param in params:
    if len(param.strip()) > 0:
      key, val = param.split()
      ret[key] = val
  return ret

if __name__ == "__main__":
  # 从输入参数中获取输入文件的路径以及输出文件的路径
  params = getParams(sys.argv[1:])
  if 'i' in params.keys():
    FILE_DIR = params['i']
  if 'o' in params.keys():
    IMG_DIR = params['o']
  if 'tm' in params.keys():
    THREAD_AMOUNT = int(params['tm'])
  if not os.path.exists(FILE_DIR):
    logger('ERROR', 'Input dir is not exists!')
    exit(-1)
  logger('INFO', 'Initializing')
  # 清空日志文件
  open(LOG_DIR, 'w').close()
  # 如果输出路径不存在则创建
  if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)
  logger('INFO', 'Input dir is %s' % FILE_DIR)
  logger('INFO', 'Output dir is %s' % IMG_DIR)
  logger('INFO', 'Total %d threads for downloading' % THREAD_AMOUNT)
  logger('INFO', 'Get all files in the input dir and create the corresponding output folder...')
  fileList = []
  for root, dirs, files in os.walk(FILE_DIR):
    for fileName in files:
      fn = fileName.replace('.txt', '')
      fileList.append(fn)
      if not os.path.exists(os.path.join(IMG_DIR, fn)):
        os.makedirs(os.path.join(IMG_DIR, fn))
  logger('INFO', 'Total input file: %d' % len(fileList))
  logger('INFO', 'Waiting for the downloading URLs..')
  for fn in fileList:
    fp = os.path.join(FILE_DIR, fn + ".txt")
    _file = open(fp, 'r')
    line = _file.readline()
    while line:
      TOTAL_AMOUNT += 1
      line = _file.readline()
  logger('INFO', 'Total URLs for downloading %d ' % TOTAL_AMOUNT)
  logger('INFO', 'Split the file lists into %d batches and delcare the same downloader' % THREAD_AMOUNT)
  sys.stdout.write('\r[INFO]\t0/' + str(TOTAL_AMOUNT) + 'complete 0.00%, average download speed ---kb/s, consume 0秒')
  sys.stdout.flush()
  pool = []
  diff = len(fileList) / THREAD_AMOUNT
  start = 0
  end = 0
  # 申明downloader
  for i in xrange(0, THREAD_AMOUNT):
    start = end
    end = start + diff
    if end > len(fileList):
      end = len(fileList)
    pool.append(BatchDownloader('downloader %d' % i, fileList[start : end]))
    pool[-1].setDaemon(True)
    pool[-1].start()
  # 开始计时
  global timer
  timer.setDaemon(True)
  timer.start()
  try:
    while FINISH_THREAD_AMOUNT[0] < THREAD_AMOUNT:
      time.sleep(5)
      pass
  except KeyboardInterrupt:
    logger('\nWARN', 'force to quit the main threads')
    sys.exit()
  # for downloader in pool:
  #   downloader.join()
  logger('\nINFO', 'Mission accomplishment')
