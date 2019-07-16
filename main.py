# -*- coding: utf-8 -*-
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 120  # 分辨率
plt.rcParams['font.sans-serif'] = ['Simhei']


def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]  # height(rows) of image
    cols1 = img1.shape[1]  # width(colums) of image
    # shape[2]#the pixels value is made up of three primary colors
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    # 初始化输出的新图像，将两幅实验图像拼接在一起，便于画出特征点匹配对
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    out[:rows1, :cols1] = np.dstack([img1])  # Python切片特性，初始化out中img1，img2的部分
    out[:rows2, cols1:] = np.dstack([img2])  # dstack,对array的depth方向加深
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv.circle(out, (int(x1), int(y1)), 5, (255, 255, 0), 1)  # 蓝绿色点，半径是4
        cv.circle(out, (int(x2) + cols1, int(y2)), 5, (0, 255, 255), 1)  # 绿加红得黄色点
        cv.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)  # 蓝色连线

    return out


def main():
    # =============================================================================
    # 原图
    # =============================================================================
    img = cv.imread("OriginalImage.png")
    gray = cv.imread("OriginalImage.png", cv.IMREAD_GRAYSCALE)
    print("图片大小: ", gray.shape)
    print("像素总数: ", gray.size)
    print("编码方式: ", gray.dtype)
    print()
    plt.subplot(1,4,1)
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.title("Original Image")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


    # =============================================================================
    # 创建fast特征点检测器
    # =============================================================================
    fast = cv.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    print("检测示例阈值: ", fast.getThreshold())
    print("检测示例邻域: ", fast.getType())
    print()
    # =============================================================================
    # 使用非最大值抑制检测
    # =============================================================================
    KeyPoints1 = fast.detect(gray, None)  # FAST特征点检测到的坐标结果
    KeyPoints1, des1 = brief.compute(img, KeyPoints1)
    img2 = cv.drawKeypoints(img, KeyPoints1, gray, color=(255, 0, 0))
    # 特征点检测参数
    print("是否为非最大值抑制: ", fast.getNonmaxSuppression())
    print("关键点个数: ", len(KeyPoints1))
    print(des1)
    print()
    # 显示使用非最大值抑制特征检测效果
    plt.subplot(1, 4, 2)
    plt.imshow(img2[:, :, [2, 1, 0]])
    plt.title("with_nonmaxSuppression")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    cv.imwrite("with_nonmaxSuppression.png", img2)
    # =============================================================================
    # 不使用非最大值抑制
    # =============================================================================
    fast.setNonmaxSuppression(0)
    KeyPoints2 = fast.detect(img, None)
    KeyPoints2, des2 = brief.compute(img, KeyPoints2)
    img3 = cv.drawKeypoints(img, KeyPoints2, gray, color=(255, 0, 0))
    # 特征点检测参数
    print("是否为非最大值抑制: ", fast.getNonmaxSuppression())
    print("关键点个数: ", len(KeyPoints2))
    print(des2)
    print()
    # 显示不使用非最大值抑制特征检测效果
    plt.subplot(1, 4, 3)
    plt.imshow(img3[:, :, [2, 1, 0]])
    plt.title("without_nonmaxSuppression")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    cv.imwrite("without_nonmaxSuppression.png", img3)
    # =============================================================================
    # 使用ORB的特征检测
    # =============================================================================
    orb = cv.ORB_create()
    # find the keypoints with ORB
    KeyPoints3 = orb.detect(img, None)
    # compute the descriptors with ORB
    KeyPoints3, des3 = orb.compute(img, KeyPoints3)
    # draw only keypoints location,not size and orientation
    img4 = cv.drawKeypoints(img, KeyPoints3, gray, color=(255, 0, 0))
    plt.subplot(1, 4, 4)
    plt.imshow(img4[:, :, [2, 1, 0]])
    plt.title('oFAST')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()
    # =============================================================================
    # 特征值描述匹配（旋转90度）
    # =============================================================================
    img0 = cv.imread('rotatedOriginalImage.png')
    gray0 = cv.imread('rotatedOriginalImage.png', cv.IMREAD_GRAYSCALE)
    fast = cv.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
    KeyPoints01 = fast.detect(img0, None)  # FAST特征点检测到的坐标结果
    KeyPoints01, des01 = brief.compute(img0, KeyPoints01)
    img01 = cv.drawKeypoints(img0, KeyPoints01, gray0, color=(255, 0, 0))
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # 暴力匹配
    matches = bf.match(des1, des01)
    out1 = drawMatches(img, KeyPoints1, img0, KeyPoints01, matches)
    plt.imshow(out1[:, :, [2, 1, 0]])
    plt.title('FAST(with_nonmaxSuppression)&BRIEF')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

    fast.setNonmaxSuppression(0)
    KeyPoints02 = fast.detect(img0, None)
    KeyPoints02, des02 = brief.compute(img0, KeyPoints02)
    img02 = cv.drawKeypoints(img0, KeyPoints02, gray0, color=(255, 0, 0))
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # 暴力匹配
    matches = bf.match(des2, des02)
    out2 = drawMatches(img, KeyPoints2, img0, KeyPoints02, matches)
    plt.imshow(out2[:, :, [2, 1, 0]])
    plt.title('FAST(without_nonmaxSuppression)&BRIEF')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

    # find the keypoints with ORB
    KeyPoints03 = orb.detect(img0, None)
    # compute the descriptors with ORB
    KeyPoints03, des03 = orb.compute(img0, KeyPoints03)
    # draw only keypoints location,not size and orientation
    img03 = cv.drawKeypoints(img0, KeyPoints03, gray0, color=(255, 0, 0))
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # 暴力匹配
    matches = bf.match(des3, des03)
    out3 = drawMatches(img, KeyPoints3, img0, KeyPoints03, matches)
    plt.imshow(out3[:, :, [2, 1, 0]])
    plt.title('ORB(oFAST&rBRIEF)')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

    # =============================================================================
    # 算法参数对关键点个数的影响
    # =============================================================================
    ThreArr = np.linspace(10, 100, 50)  # 阈值取值数组
    NumWith = np.zeros(len(ThreArr))  # 使用非最大值抑制的关键点个数数组
    NumWithout = np.zeros(len(ThreArr))  # 不使用非最大值抑制的关键点个数数组

    # 5/8邻域
    fast = cv.FastFeatureDetector_create(type=cv.FAST_FEATURE_DETECTOR_TYPE_5_8)  # 设置邻域
    count = 0  # 计数变量
    for thre in ThreArr:
        fast.setThreshold(int(thre))  # 设置阈值
        count += 1
        # 使用非最大值抑制
        fast.setNonmaxSuppression(1)
        KeyPoints = fast.detect(img, None)
        NumWith[count - 1] = len(KeyPoints)
        # 不使用非最大值抑制
        fast.setNonmaxSuppression(0)
        KeyPoints = fast.detect(img, None)
        NumWithout[count - 1] = len(KeyPoints)
    plt.plot(ThreArr, NumWith, label="使用非最大值抑制")
    plt.plot(ThreArr, NumWithout, label="不使用非最大值抑制")
    plt.legend()
    plt.xlabel("检测阈值")
    plt.ylabel("关键点个数")
    plt.grid()
    plt.title("5/8邻域")
    plt.show()

    # 7/12邻域
    fast = cv.FastFeatureDetector_create(type=cv.FAST_FEATURE_DETECTOR_TYPE_7_12)  # 设置邻域
    count = 0  # 计数变量
    for thre in ThreArr:
        fast.setThreshold(int(thre))  # 设置阈值
        count += 1
        # 使用非最大值抑制
        fast.setNonmaxSuppression(1)
        KeyPoints = fast.detect(img, None)
        NumWith[count - 1] = len(KeyPoints)
        # 不使用非最大值抑制
        fast.setNonmaxSuppression(0)
        KeyPoints = fast.detect(img, None)
        NumWithout[count - 1] = len(KeyPoints)
    plt.plot(ThreArr, NumWith, label="使用非最大值抑制")
    plt.plot(ThreArr, NumWithout, label="不使用非最大值抑制")
    plt.legend()
    plt.xlabel("检测阈值")
    plt.ylabel("关键点个数")
    plt.grid()
    plt.title("7/12邻域")
    plt.show()

    # 9/16邻域
    fast = cv.FastFeatureDetector_create(type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16)  # 设置邻域
    count = 0  # 计数变量
    for thre in ThreArr:
        fast.setThreshold(int(thre))  # 设置阈值
        count += 1
        # 使用非最大值抑制
        fast.setNonmaxSuppression(1)
        KeyPoints = fast.detect(img, None)
        NumWith[count - 1] = len(KeyPoints)
        # 不使用非最大值抑制
        fast.setNonmaxSuppression(0)
        KeyPoints = fast.detect(img, None)
        NumWithout[count - 1] = len(KeyPoints)
    plt.plot(ThreArr, NumWith, label="使用非最大值抑制")
    plt.plot(ThreArr, NumWithout, label="不使用非最大值抑制")
    plt.legend()
    plt.xlabel("检测阈值")
    plt.ylabel("关键点个数")
    plt.grid()
    plt.title("9/16邻域")
    plt.show()


if __name__ == '__main__':
    main()
