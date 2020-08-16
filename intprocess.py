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
from itertools import groupby
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
import strprocess

#得分点
class intPoint:
    def __init__(self, HTML,HtmlParser,stringOne):
        self.HTML = HTML
        self.HtmlParser = HtmlParser
        self.group = []
        self.value = []
        self.maxvalue = []
        self.minvalue = []
        self.scorevalue = []
        self.counter  = 0
        self.maxflag = 0
        self.stringOne = stringOne
        self.typelist = [] #记载类型
        self.typeaddr = [] #记载相对位置
        self.maxrange = []
        self.minrange = []
        self.minindex = 0
        self.maxindex = 0
    #计分使用多维度的最小二乘算法来计算
    def int2float_analyze(self):
        
        data = self.HtmlParser.split_str(self.HtmlParser.data)
        self.HTML.write_html("<br>")
        self.HTML.create_table()
        self.HTML.start_tr()
        self.HTML.add_td("<p>" + "待检测字符:" + self.stringOne.list_to_string(data) + "</p>")
        self.HTML.end_tr()
        self.maxflag  = len(data)
        self.maxindex = len(data)
        self.stringOne.int2float()
        #求解int/float在str中的相对位置，并结合string的位置确定出flaot的相对位置再次缩小范围
        self.HTML.start_tr()
        self.HTML.add_td("<p>" + "当前计数"   + "</p>")
        self.HTML.add_td("<p>" + "最小索引" + "</p>")
        self.HTML.add_td("<p>" + "最大索引" + "</p>")
        self.HTML.add_td("<p>" + "索引字符" + "</p>")
        self.HTML.end_tr()
        counter = 0
        #前向
        for ii in self.typelist:
            if( ii == "str" ):
#                 pdb.set_trace()
#                 存储当前的索引
                self.minindex = self.stringOne.maxrange[counter]
                counter = counter + 1
            elif( ii == "int" ):
                # str[] int[] str[]
                #前一个的最大值是int的最小值，后一个的最小值是int 的最大值
                self.minrange.append(self.minindex)
        counter = 0
        #后向
        for ii in self.typelist[::-1]:
            if( ii == "str" ):
#                 pdb.set_trace()
#                 存储当前的索引
                self.maxindex = self.stringOne.minrange[::-1][counter]
                counter = counter + 1
            elif( ii == "int" ):
                # str[] int[] str[]
                #前一个的最大值是int的最小值，后一个的最小值是int 的最大值
                self.maxrange.append(self.maxindex)
        self.maxrange = self.maxrange[::-1]
        counter = 0
        for ii in self.minrange:
            self.HTML.start_tr()
            self.HTML.add_td("<p>" + str(counter) + "</p>")
            self.HTML.add_td("<p>" + str(self.minrange[counter]) + "</p>")
            self.HTML.add_td("<p>" + str(self.maxrange[counter]) + "</p>")
            if(self.minrange[counter] == self.maxrange[counter]):
                self.HTML.add_td("<p>" + self.stringOne.list_to_string(data[self.minrange[counter]]) + "</p>")
            else:
                self.HTML.add_td("<p>" + self.stringOne.list_to_string(data[self.minrange[counter]:self.maxrange[counter]]) + "</p>")
            self.HTML.end_tr()
            counter = counter + 1
        #再次缩小范围
        for ii in range(1,len(self.minrange),1):
            if(self.minrange[ii] < self.minrange[ii-1] ):
#                 pdb.set_trace()
                self.minrange[ii] = self.minrange[ii-1]
        #倒序
        for xx in range(len(self.maxrange)-1,0,-1):
#             pdb.set_trace()
            if(self.maxrange[xx-1] > self.maxrange[xx] ):
                self.maxrange[xx-1] = self.maxrange[xx]
        #开始计算，是否能够正确的匹配个数，否则默认设置为它的最大值
        counter = 0
        self.HTML.end_table()
        self.HTML.write_html("<br>")
        self.HTML.create_table()
        
        for vv in self.maxvalue:
            #非小数点和空格分割全部的字符串，data 也同样分割
            digit_list = []
            for dd in data[self.minrange[counter]:self.maxrange[counter]+1] :
                digit_list.append(self.intnumber_search(dd) )
            counter = counter + 1
            #每一个最大最小范围包含一个列表，后处理将使用子集对维度空间计算方式
            self.value.append(digit_list)
            self.HTML.start_tr()
            self.HTML.add_td("<p>" + str(counter) + "</p>")
            self.HTML.add_td("<p>" + self.stringOne.list_to_string(digit_list) + "</p>")
            self.HTML.end_tr()
        self.HTML.end_table()
        
        self.HTML.write_html("<br>")
        self.HTML.create_table()
        self.HTML.start_tr()
        self.HTML.add_td("<p>" + "jupyter中检测到的字符:" + "<br>" + self.stringOne.list_to_string(self.value) + "</p>")
        self.HTML.end_tr()
        self.HTML.end_table()
    def intnumber_search(self,data):
        digit_list = []
        float_list = [''.join(list(g)) for k, g in groupby(data, key=lambda x: x.isdigit())]
        counter = 0
        for ff in float_list:
            if ff.isdigit() :
                if (   ( (counter+2) <= (len(float_list)-1))  and \
                       (float_list[counter+2].isdigit()) and \
                       (float_list[counter+1] == ".") ):
                    digit_list.append(ff+"."+float_list[counter+2])
                    float_list[counter+2] = "null"
                    float_list[counter+1] = "null"
                else:
                    digit_list.append(ff)
            counter = counter + 1
        return digit_list
        
        