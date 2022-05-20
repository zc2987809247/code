import math

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
    cv_show(noisy_img)
    cv.imwrite('noisy-img.jpg', noisy_img)


# 去除噪声，滤波函数
def remove_noise(img):
    # # 均值滤波,区附近3*3的像素值取平均,
    # blur = cv.blur(cv.imread('noisy-img.jpg', cv.IMREAD_COLOR), (3, 3))
    #
    # # 方框滤波，级别基本和均值一样,效果不错，-1表示颜色通道一致
    # box = cv.boxFilter(cv.imread('noisy-img.jpg', cv.IMREAD_COLOR), -1, (3, 3), normalize=True)
    #
    # 高斯滤波
    aussian = cv.GaussianBlur(img, (5, 5), 1)

    # 中值滤波,取中间数,效果不错，主要去除椒噪声和斑点噪声
    median = cv.medianBlur(img, 5)
    return aussian


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


# 绘制直方图
def histogram(img):
    histb = cv.calcHist([img], [0], None, [256], [0, 255])
    plt.plot(histb, color='b')
    plt.show()


def compute_threshold(img):
    # 大津法阈值分割
    threshold, img_bin = cv.threshold(img, -1, 255, cv.THRESH_OTSU)

    print("最佳阈值：" + str(threshold))
    # 返回最佳阈值
    return threshold


# 绘制轮廓
def draw_contours(img, mask,contours_area):
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    # 二值化处理
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    # contours为轮廓的数据集list
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    new_contours = []
    for i in range(len(contours)):
        area = cv.contourArea(contours[i], False)
        if area > contours_area:
            new_contours.append(contours[i])

    # 绘制轮廓
    draw_img = img.copy()
    # -1表示全部轮廓，（255，0，0）以BGR颜色绘制，2表示线条厚度
    res = cv.drawContours(draw_img, new_contours, -1, (0, 0, 255), 1)
    return res


def draw_defect(img, x, y, w, h, raster_num,contours_area):
    # print(raster_num, type(raster_num))
    if raster_num == 4:
        img_flag = cv.imread('img_model.jpg', cv.IMREAD_COLOR)
        # print(img_flag.shape)
    elif raster_num == 3:
        img_flag = cv.imread('img_model2.jpg', cv.IMREAD_COLOR)
    else:
        print("ERROR!")

    img_flag = cv.resize(img_flag, (w - 20, h))
    img1 = img[y:y + h, x + 20:x + w]
    # img_flag = img_flag[0:800, 20:800]
    # 转换为灰度图
    gray = cv.cvtColor(img1, cv.COLOR_BGRA2GRAY)
    gray_flag = cv.cvtColor(img_flag, cv.COLOR_BGRA2GRAY)

    x = compute_threshold(gray)
    # 二值化处理,阈值220对于污渍检测效果号，阈值170对于裂纹检测效果好
    ret, thresh = cv.threshold(gray, x, 255, cv.THRESH_BINARY)
    ret_flag, thresh_flag = cv.threshold(gray_flag, 127, 255, cv.THRESH_BINARY)
    # 去除横线部分
    thresh = ~thresh
    thresh_flag = ~thresh_flag

    # 卷积核
    kernel = np.ones((3, 3), dtype=np.uint8)
    # 对模板图进行膨胀操作
    thresh_flag = cv.dilate(thresh_flag, kernel, 1)
    # add_img = np.hstack([thresh, thresh_flag])
    # cv_show(add_img)
    ret = thresh - thresh_flag
    ret = ~ret

    # 对ret进行闭运算，先膨胀在腐蚀 mask是处理之后的缺陷图
    mask = cv.morphologyEx(ret, cv.MORPH_CLOSE, kernel)
    # 绘制出处理之后的结果图的轮廓
    return draw_contours(img1, mask,contours_area)




def upload(filepath, horizontal_num, vertical_num, raster_num,contours_area):
    # 读取图片，cv.IMREAD_COLOR,转换成3通道RGB颜色
    img = cv.imread(filepath, cv.IMREAD_COLOR)
    # h,w表示图片的高度和宽度
    h = img.shape[0]
    w = img.shape[1]
    block_h = h // vertical_num
    block_w = w // horizontal_num
    x = 0
    y = 0

    image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    while y < h:
        while x < w:
            image[y:y + block_h, x:x + block_w - 20] = draw_defect(img, x, y, block_w, block_h, raster_num,contours_area)
            x = x + block_w
        x = 0
        y = y + block_w

    image = cv.resize(image, (1800, 960))
    cv_show(image)
    return 1



# # 读取图片，cv.IMREAD_COLOR,转换成3通道RGB颜色
# img = cv.imread('sgEL_setup_01.JPG', cv.COLOR_BGRA2GRAY)





