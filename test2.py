import glob

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# 读取视频
def readtest():
    videoname = '86659034_nb2-1-64.flv'
    capture = cv.VideoCapture(videoname)
    if capture.isOpened():
        while True:
            ret, img = capture.read()

            if not ret: break
    else:
        print("视频打开失败")


def writetest():
    videoinname = '86659034_nb2-1-64.flv'
    videooutpath = 'video_out.avi'
    capture = cv.VideoCapture(videoinname)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    writer = cv.VideoWriter(videooutpath, fourcc, 20.0, (1280, 960), True)
    if capture.isOpened():
        i = 0
        while True:
            ret,img_src = capture.read()
            if not ret:break
            img_out = cv.medianBlur(img_src, 5)
            writer.write(img_out)
            print(i)
            i = i+1
    else:
        print("视频打开失败")
    writer.release()

writetest()
