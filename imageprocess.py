import sys
import re
import pdb
import os
import shutil
import urllib.request       
#导入urllib.request库
import os.path
from html.parser import HTMLParser
import requests
import string
import re
import numpy as np
import urllib
import urllib.request as urllib2

import cv2
from sklearn.decomposition import PCA
from skimage.measure import compare_ssim
from scipy.misc import imread
from PIL import Image
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import operator
from sklearn import preprocessing
from urllib import request
import xml.sax
import html
import base64
"""
5种算法如下：值哈希算法、差值哈希算法和感知哈希算法都是值越小，相似度越高，取值为0-64，即汉明距离中，64位的hash值有多少不同
三直方图和单通道直方图的值为0-1，值越大，相似度越高
"""

def aHash(img):
    # 均值哈希算法
    # 缩放为8*8
    #img = cv2.resize(img, (8, 8))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s = s+gray[i, j]
    # 求平均灰度
    avg = s/64
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str
 

def dHash(img):
    # 差值哈希算法
    # 缩放8*8
    #img = cv2.resize(img, (9, 8))
    # 转换灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str
 

def pHash(img):
    # 感知哈希算法
    # 缩放32*32
    img = cv2.resize(img, (int(img.shape[0]/2)*2, int(img.shape[1]/2)*2))   # , interpolation=cv2.INTER_CUBIC
 
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换  
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]
 
    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if max(hist1[i], hist2[i]) != 0 :
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
    degree = degree / len(hist1)
    return degree[0]


def classify_hist_with_split(image1, image2, size=(256, 256)):
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    #image1 = cv2.resize(image1, size)
    #image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    #pdb.set_trace()
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    #返回一个浮点数据
    return sub_data


def cmpHash(hash1, hash2):
    # Hash值对比
    # 算法中1和0顺序组合起来的即是图片的指纹hash。顺序不固定，但是比较的时候必须是相同的顺序。
    # 对比两幅图的指纹，计算汉明距离，即两个64位的hash值有多少是不一样的，不同的位数越小，图片越相似
    # 汉明距离：一组二进制数据变成另一组数据所需要的步骤，可以衡量两图的差异，汉明距离越小，则相似度越高。汉明距离为0，即两张图片完全一样
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


def getImageByUrl(url):
    # 根据图片url 获取图片对象
    html = requests.get(url, verify=False)
    image = Image.open(BytesIO(html.content))
    return image


def PILImageToCV():
    # PIL Image转换成OpenCV格式
    path = "/Users/waldenz/Documents/Work/doc/TestImages/t3.png"
    img = Image.open(path)
    plt.subplot(121)
    plt.imshow(img)
    print(isinstance(img, np.ndarray))
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    print(isinstance(img, np.ndarray))
    plt.subplot(122)
    plt.imshow(img)
    plt.show()

def CVImageToPIL():
    # OpenCV图片转换为PIL image
    path = "/Users/waldenz/Documents/Work/doc/TestImages/t3.png"
    img = cv2.imread(path)
    # cv2.imshow("OpenCV",img)
    plt.subplot(121)
    plt.imshow(img)
 
    img2 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()

def bytes_to_cvimage(filebytes):
    # 图片字节流转换为cv image
    image = Image.open(filebytes)
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return img

def runAllImageSimilaryFun(img1,img2):
    # 均值、差值、感知哈希算法三种算法值越小，则越相似,相同图片值为0
    # 三直方图算法和单通道的直方图 0-1之间，值越大，越相似。 相同图片为1
 
    # t1,t2   14;19;10;  0.70;0.75
    # t1,t3   39 33 18   0.58 0.49
    # s1,s2  7 23 11     0.83 0.86  挺相似的图片
    # c1,c2  11 29 17    0.30 0.31
    
    hash1 = aHash(img1)
    hash2 = aHash(img2)
    n1 = cmpHash(hash1, hash2)
 
    hash1 = dHash(img1)
    hash2 = dHash(img2)
    n2 = cmpHash(hash1, hash2)
    
    hash1 = pHash(img1)
    hash2 = pHash(img2)
    n3 = cmpHash(hash1, hash2)
    
    #pdb.set_trace()
    n4 = classify_hist_with_split(img1, img2)
    
    n5 = calculate(img1, img2)
 
    plt.subplot(121)
    plt.imshow(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)))
    plt.subplot(122)
    plt.imshow(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)))
    plt.show()
    
    return(n1,n2,n3,n4,n5)

# TODO 实现以下函数并输出所选直线的MSE
def calculateMSE(X,Y):
    #pdb.set_trace()
    in_bracket = []
    for i in range(len(X)):
        num = Y[i] - X[i]
        num = pow(num,2)
        in_bracket.append(num)

    all_sum = sum(in_bracket)
    MSE = all_sum / len(X)
    return MSE

def picture_process(image1,image2):

    img = image1
    min_size = min(img.shape[0],img.shape[1])
    if(min_size > 100):
        #等比例缩放
        if( min_size == img.shape[0] ):
            img_re = cv2.resize(img, ( int(img.shape[1] * 100 / img.shape[0] ) , 100) )
        else:
            img_re = cv2.resize(img, ( 100 , int(img.shape[0] * 100 / img.shape[1] ) ) )
    #cv2.imwrite(name,img_re)
    fft2 = np.fft.fft2(img_re)
    shift2center = np.fft.fftshift(fft2)
    plt.subplot(234),plt.imshow(np.abs(shift2center)/100,'gray'),plt.title('shift2center')
    #print(shift2center/100)
    #对中心化后的结果进行对数变换，并显示结果
    log_shift2center = np.log(1 + np.abs(shift2center))
    plt.subplot(236),plt.imshow(log_shift2center,'gray'),plt.title('log_shift2center')
    plt.show()
    #图像的傅里叶变换得到后开始计算他们之间的差别，并作为得分项目
    #矩阵主成分的提取
    pca_img_1 = PCA(n_components=1).fit_transform(log_shift2center)
    #a_img_1 = np.sum(pca_img_1, axis=1) 
    #标准数据也做一遍同样的处理

    
    img = image2

    min_size = min(img.shape[0],img.shape[1])

    if(min_size > 100):
        #等比例缩放
        if( min_size == img.shape[0] ):
            img_re = cv2.resize(img, ( int(img.shape[1] * 100 / img.shape[0] ) , 100) )
        else:
            img_re = cv2.resize(img, ( 100 , int(img.shape[0] * 100 / img.shape[1] ) ) )

        #cv2.imwrite(name,img_re)
    fft2 = np.fft.fft2(img_re)
    shift2center = np.fft.fftshift(fft2)
    #print(shift2center/100)
    #对中心化后的结果进行对数变换，并显示结果
    log_shift2center = np.log(1 + np.abs(shift2center))
    #图像的傅里叶变换得到后开始计算他们之间的差别，并作为得分项目
    #矩阵主成分的提取
    pca_img_2 = PCA(n_components=1).fit_transform(log_shift2center)
    #a_img_2 = np.sum(pca_img_2, axis=1) 
    return(pca_img_1,pca_img_2)

#得分点
class PicturesPoint:
    def __init__(self, HTML,HtmlParser,template_path):
        self.HTML = HTML
        self.HtmlParser = HtmlParser
        self.num = self.HtmlParser.downImgs(HtmlParser.get_pngs())
        self.attribute = ["均方误差","SSIM结构相似性度量","均值哈希算法相似度aHash","差值哈希算法相似度dHash","感知哈希算法相似度Hash","三直方图算法相似度","单通道的直方图"]
        #打分采用等级形式,待识别的等级，值越小等级越高
        self.attr_value = np.zeros( [self.num,(len(self.attribute))] )
        self.group = []
        self.value = []
        self.scorevalue = []
        self.counter  = 0
        self.min_value = [1, 0, 1, 1, 1, 0, 0]
        self.max_value = [0, 1, 0, 0, 0, 1, 1]
        self.degree = [1,1,2,2,2,3,3]
        #处一统计级别的打分规则，满分最高分
        self.degree_score = [100,80,60,50,40,20,0]
        #同一等级内的将采用权重的方式来划分
        self.weights = [35,35,7,7,7,9,9]
        self.template_path = template_path
    def all(self):
        self.HTML.create_table()
        self.HTML.start_tr()
        self.HTML.add_td("<p>" + "图片编号:" + "</p>")
        self.HTML.add_td("<p>" + "均方误差:" + "</p>")
        self.HTML.add_td("<p>" + "SSIM结构相似性度量为：" + "</p>")
        self.HTML.add_td("<p>" + "均值哈希算法相似度aHash：" + "</p>")
        self.HTML.add_td("<p>" + "差值哈希算法相似度dHash：" + "</p>")
        self.HTML.add_td("<p>" + "感知哈希算法相似度pHash：" +  "</p>")
        self.HTML.add_td("<p>" + "三直方图算法相似度：" + "</p>")
        self.HTML.add_td("<p>" + "单通道的直方图" +  "</p>")
        self.HTML.add_td("<p>" + "模板图片" +  "</p>")
        self.HTML.add_td("<p>" + "待匹配图片" +  "</p>")
        self.HTML.end_tr()
        for kk in self.value:
            self.processimage()
            self.counter = self.counter + 1
#         grade = self.get_point()
        self.HTML.end_table()
#         return grade
    def processimage(self):
        #attribute,degree,degree_score,weights,min_value,max_value = ScoringRules()
        print("<========================第%d张图片===========================>>\n"%(self.counter+1))
        image_path1 = "/temp/" + str(self.counter) + ".png"
        image_path2 = self.template_path + "/" + self.value[self.counter]
        #检测图像是否存在
        if( os.path.exists(image_path1) and os.path.exists(image_path2 ) ):
            img1_src  = cv2.imread(image_path1)
            img1_gray = 0.2126 * img1_src[:,:,0] + 0.7152 * img1_src[:,:,1] + 0.0722 * img1_src[:,:,2]
            img2_src  = cv2.imread(image_path2)
            img2_gray = 0.2126 * img2_src[:,:,0] + 0.7152 * img2_src[:,:,1] + 0.0722 * img2_src[:,:,2]
            pca_img_1,pca_img_2 = picture_process(img1_gray,img2_gray)
            #首先需要归一化处理，如下所示
            #pdb.set_trace()
            MSE = calculateMSE(pca_img_1,pca_img_2)
            #img = cv2.imread(image_path1)
            #img1 = imageio.imread(image_path1)
            #img2 = imageio.imread(image_path2)
            img2_src = np.resize(img2_src, (img1_src.shape[0], img1_src.shape[1], img1_src.shape[2]))
            ssim = compare_ssim(img1_src,img2_src, multichannel=True)
            n1,n2,n3,n4,n5 = runAllImageSimilaryFun(img1_src,img2_src)
            self.HTML.start_tr()
            self.HTML.add_td("<p>" + str(self.counter+1) + "</p>")
            self.HTML.add_td("<p>" + str(MSE) + "</p>")
            self.HTML.add_td("<p>" + str(ssim) + "</p>")
            self.HTML.add_td("<p>" + str(n1) + "</p>")
            self.HTML.add_td("<p>" + str(n2) + "</p>")
            self.HTML.add_td("<p>" + str(n3) +  "</p>")
            self.HTML.add_td("<p>" + str(n4) + "</p>")
            self.HTML.add_td("<p>" + str(n5) +  "</p>")
            f = open(image_path1,"rb")
            image_base64 = str(base64.b64encode(f.read()), encoding='utf-8')
            self.HTML.add_td("""<img src="data:image/png;base64,""" + image_base64 +  """""/>""")
            f = open(image_path1,"rb")
            image_base64 = str(base64.b64encode(f.read()), encoding='utf-8')
            self.HTML.add_td("""<img src="data:image/png;base64,""" + image_base64 +  """""/>""")
            self.HTML.end_tr()
        else:
            [MSE,ssim,n1,n2,n3,n4,n5] = self.min_value
            self.HTML.start_tr()
            self.HTML.add_td("<p>" + str(self.counter+1) + "</p>")
            self.HTML.add_td("<p>" + str(MSE) + "</p>")
            self.HTML.add_td("<p>" + str(ssim) + "</p>")
            self.HTML.add_td("<p>" + str(n1) + "</p>")
            self.HTML.add_td("<p>" + str(n2) + "</p>")
            self.HTML.add_td("<p>" + str(n3) +  "</p>")
            self.HTML.add_td("<p>" + str(n4) + "</p>")
            self.HTML.add_td("<p>" + str(n5) +  "</p>")
            self.HTML.end_tr()
        self.attr_value[self.counter,:] = [MSE,ssim,n1,n2,n3,n4,n5]

        
#         #pdb.set_trace()
#         print("均方误差：%.4f"%MSE)
#         print("SSIM结构相似性度量为%.3f"%ssim)
#         #对于大于0的图片
#         print('均值哈希算法相似度aHash：', n1)
#         print('差值哈希算法相似度dHash：', n2)
#         print('感知哈希算法相似度pHash：', n3)
#         print('三直方图算法相似度：', n4)
#         print("单通道的直方图", n5)
#         print("%d %d %d %.2f %.2f " % (n1, n2, n3, round(n4, 2), n5))
#         print("%.2f %.2f %.2f %.2f %.2f " % (1-float(n1/64), 1 -
#                                          float(n2/64), 1-float(n3/64), round(n4, 2), n5))
#         #pdb.set_trace()
#         print("just")
#         print(self.attr_value[self.counter,:])
        
        
    