import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# 显示图像
def cv_show(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 给图片增加噪音
def add_noise(img):
    mean = 0
    # 设置高斯分布的标准差
    sigma = 25
    # 根据均值和标准差生成符合高斯分布的噪声,shape会返回tuple元组，第一个元素表示矩阵行数，第二个元组表示矩阵列数，
    gauss = np.random.normal(mean, sigma, (img.shape[0], img.shape[1], 3))
    # 给图片添加高斯噪声
    noisy_img = img + gauss
    # 设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img, a_min=0, a_max=img.shape[0] * img.shape[1])
    # 保存图片
    cv.imwrite('noisy-img.jpg', noisy_img)


# 去除噪声，滤波函数
def remove_noise(img):
    # 均值滤波,区附近3*3的像素值取平均,
    blur = cv.blur(cv.imread('noisy-img.jpg', cv.IMREAD_COLOR), (3, 3))

    # 方框滤波，级别基本和均值一样,效果不错，-1表示颜色通道一致
    box = cv.boxFilter(cv.imread('noisy-img.jpg', cv.IMREAD_COLOR), -1, (3, 3), normalize=True)

    # 高斯滤波
    aussian = cv.GaussianBlur(cv.imread('noisy-img.jpg', cv.IMREAD_COLOR), (5, 5), 1)

    # 中值滤波,取中间数,效果不错，主要去除椒噪声和斑点噪声
    median = cv.medianBlur(img, 5)
    return median


# 边缘检测
def edge_detection(img):
    # sobel算子
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    sobelx = cv.convertScaleAbs(sobelx)
    sobely = cv.convertScaleAbs(sobely)
    sobelxy = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    # scharr算子
    scharrx = cv.Scharr(img, cv.CV_64F, 1, 0)
    scharry = cv.Scharr(img, cv.CV_64F, 0, 1)
    scharrx = cv.convertScaleAbs(sobelx)
    scharry = cv.convertScaleAbs(sobely)
    scharrxy = cv.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

    # laplacian算子
    laplacian = cv.Laplacian(img, cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian)

    res = np.hstack((sobelxy, scharrxy, laplacian))
    cv_show(res)


# 边缘检测
def cv_canny(img):
    # 去噪
    img = remove_noise(img)
    # canny边缘检测
    v1 = cv.Canny(img, 100, 150)
    return v1


# 读取图片，cv.IMREAD_COLOR,转换成3通道RGB颜色
img = cv.imread('sgEL_setup_06_2.jpg.jpg', cv.IMREAD_COLOR)
ret,thresh = cv.threshold(img,127,255,cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

draw_img = img.copy()
res = cv.drawContours(draw_img, contours, -1, (0, 0, 255, 2))
cv_show(res)
