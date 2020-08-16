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
#得分点
class stringPoint:
    def __init__(self, HTML,HtmlParser):
        self.HTML = HTML
        self.HtmlParser = HtmlParser
        self.group = []
        self.value = []
        self.scorevalue = []
        self.counter  = 0
        self.minrange = []
        self.maxrange = []
        self.minflag = 0
        self.maxflag = 0
        self.score = []
    
    
    def list_to_string(self,list_data):
        string_line = ""
        for ll in list_data:
            
            if( isinstance(ll,int) or isinstance(ll,float) ):
                string_line = string_line + "*" + str(ll)
            elif( isinstance(ll,str) ):
                string_line = string_line + "*" + ll
            elif( isinstance(ll,list) ):
                #list则递归调用
                string_line= string_line + self.list_to_string(ll)
            else:
                pdb.set_trace()
                string_line = string_line + " " + "{null}"
        string_line = string_line.replace("<", "{")
        string_line = string_line.replace(">", "}")
        return(string_line)
    
    def strmatching(self):
        #字符打乱，使用模糊匹配的方法
        #1、先寻找唯一能匹配到的字符串
        #2、每个字符串添加一个位置参数，这个参数和xml文件中的对应
        #3、每个字符串添加一个范围参数，更具1来确定的
        #4、如果1 不能 匹配到唯一的一个字符串，那么我们就寻找开始位置最连贯的哪个字符串
        #5、目的是为了尽可能的缩小搜索范围，每个字符串，然后在这个里面搜索，看能不能匹配到
        #6、如果匹配到就记分，否则就记为缺失项
        data = self.HtmlParser.split_str(self.HtmlParser.data)
        self.maxflag = len(data) - 1
        for kk in self.value:
            if(data.count(kk) == 1):
                self.minrange.append(data.index( kk ))
                self.maxrange.append(data.index( kk ))
                self.minflag = data.index( kk )
            else:
                self.minrange.append(self.minflag)
                self.maxrange.append(self.maxflag)
        #需要重新确定下最大的索引
        #正序
        for ii in range(1,len(self.minrange),1):
            if(self.minrange[ii] < self.minrange[ii-1] ):
#                 pdb.set_trace()
                self.minrange[ii] = self.minrange[ii-1]
        #倒序
        for xx in range(len(self.maxrange)-1,0,-1):
#             pdb.set_trace()
            if(self.maxrange[xx-1] > self.maxrange[xx] ):
                self.maxrange[xx-1] = self.maxrange[xx]
        self.HTML.write_html("<br>")
        self.HTML.create_table()
        self.HTML.start_tr()
        print(data)
        self.HTML.add_td("<p>" + "jupyter中检测到的字符:" + "<br>" + self.list_to_string(data) + "</p>")
        self.HTML.add_td("<p>" + "需要被检测到的字符" + "<br>" + self.list_to_string(self.value) + "</p>")
        self.HTML.add_td("<p>" + "被检测到的字符最小范围" + "<br>" + self.list_to_string(self.minrange) + "</p>")
        self.HTML.add_td("<p>" + "被检测到的字符最大范围" + "<br>" + self.list_to_string(self.maxrange) + "</p>")
        self.HTML.end_tr()
        self.HTML.end_table()
        #在次匹配
        counter = 0
        self.HTML.write_html("<br>")
        self.HTML.create_table()
        for kk in self.value:
#             pdb.set_trace()
            if( kk in data[self.minrange[counter]:self.maxrange[counter]+1]):
                self.score.append(100)
            else:
                #如果不在搜索范围以内，则再次执行搜索
                self.score.append(self.split_str(kk,counter,data))
            counter = counter + 1
        self.HTML.end_table()
    def split_str(self,kk,counter,data):
        #非字符和数字的分割，分割后按照比例再次计算分值
        word_score = []
#         按照任何给单词分割字符串
        regEx = re.compile('\W+')
        word_split = regEx.split(kk)
        stradd = ""
        for ss in data[self.minrange[counter]:self.maxrange[counter]+1]:
            stradd = stradd + " " + ss
        self.HTML.start_tr()
        self.HTML.add_td("<p>" + str(counter) + "</p>")
        self.HTML.add_td("<p>" + " 未检测到的字符分割" + self.list_to_string(word_split) + "</p>")
        
        for kk in word_split:
            #检测是否在所在范围以内，需要注意的是，元素的部分字符串的匹配。如果存在，则
            search = re.search( kk , stradd)
            straddprint = stradd.replace("<", "{")
            straddprint =straddprint.replace(">", "}")
            self.HTML.add_td("<p>" + "未检测到的字符分割后被匹配项目" + "<br>" + straddprint + "</p>")
            if( search ):
                stradd = stradd[0:search.span()[0]]+stradd[search.span()[1]:]
                word_score.append(100)
            else:
                word_score.append(0)
        if word_score != [] :
            return sum(word_score) / len(word_score)
        else:
            return 0
        self.HTML.end_tr()

    
    #在确认的范围内，再次对未搜索到的字符串分割搜索，确认唯一的ID大小
    def split_int2float(self,kk,counter,data,data_copy):
        #非字符和数字的分割，分割后按照比例再次计算分值
        word_score = []
#         按照任何给单词分割字符串
        regEx = re.compile('\W+')
        word_split = regEx.split(kk)
        stradd = ""
        for ss in data[self.minrange[counter]:self.maxrange[counter]+1]:
            stradd = stradd + " " + ss
        for kk2 in word_split:
            #检测是否在所在范围以内，需要注意的是，元素的部分字符串的匹配。如果存在，则
            #确定下是否为唯一的索引
            if(len(re.findall(kk2, stradd))==self.value.count(kk)):
                # 再次计数
                ddcounter = 0
                
                for dd in data_copy[self.minrange[counter]:self.maxrange[counter]+1]:
                    
                    if(len(re.findall(kk2, dd))==1):
                        self.minrange[counter] = self.minrange[counter]  +  ddcounter #确定当前data的索引值在min下的偏移值
                        self.maxrange[counter] = self.minrange[counter]  #
                    
                        data_copy[self.minrange[counter]] = ""
                        return 0
                    ddcounter = ddcounter + 1
        
    def int2float(self):
        #根据字符的匹配确定一个范围
        data = self.HtmlParser.split_str(self.HtmlParser.data)
        data_copy = data.copy()
        counter = 0
        for kk in self.value:
#             pdb.set_trace()
            if( kk in data[self.minrange[counter]:self.maxrange[counter]+1]):
                pass
            else:
                #如果不在搜索范围以内，则再次执行搜索
                self.split_int2float(kk,counter,data,data_copy)
            
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
        print(self.minrange)
        print(self.maxrange)
        return([self.minrange,self.maxrange])
            
            
        
        
        