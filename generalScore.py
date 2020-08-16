#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import imageprocess
import strprocess
import intprocess
import htmlprocess
import xmlprocess
import SyntheticAnalyser
import DependenceTreeprocess


# In[2]:


if __name__ == '__main__':
    
    name="5.基于python的图像去噪实验5"
    ipynb_path="图像去噪v1/" + name  +".ipynb"
    template_path="图像去噪v1/" + name
    html_reportfile = 'ReportCard.html'
#     b=".基于pytorch复杂深度神经网络搭建"
#     a="pytorch实验/12.基于pytorch复杂深度神经网络搭建"
    #提示用户输入信息，并强制类型转换为字符串型
    html_file = "/temp/" + ipynb_path.split("/")[-1].replace(".ipynb", ".html")
    py_file = "/temp/" + ipynb_path.split("/")[-1].replace(".ipynb", ".py")
    if os.path.exists(html_file) == True:
        os.system("rm -rf " + html_file)
    os.system("jupyter nbconvert --to html --output-dir='/temp' " + ipynb_path )
    os.system("jupyter nbconvert --to python --output-dir='/temp' "+ ipynb_path )
    
    if os.path.exists(html_file) == False or os.path.exists(py_file) == False :
        print(html_file + '文件不存在!')
        sys.exit(1)
    #开始解析HTML，自动调用HTMLParser中的内置方法
    HtmlParser=htmlprocess.myHTMLParser()
    content = HtmlParser.read_file(html_file)
    HtmlParser.feed(content)
    #把实验报告写到html文件
    HTML = htmlprocess.html_display(html_reportfile)
    HTML.start()
    #初始化图像处理函数
    pictureOne = imageprocess.PicturesPoint(HTML,HtmlParser)
    stringOne = strprocess.stringPoint(HTML,HtmlParser)
    intOne = intprocess.intPoint(HTML,HtmlParser,stringOne)
    #解析下xml文件
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # 重写 ContextHandler
    Handler = xmlprocess.xmlHandler(pictureOne,stringOne,intOne)
    parser.setContentHandler( Handler )
    for f in os.listdir(template_path):
        if f.endswith('.xml'):
            xml_file = f
    parser.parse(template_path +"/"+ xml_file)
    #图像处理函数初始化完成，开始执行处理函数
    pictureOne.all()
#     print("总成绩为%.3f"%my_grade)
    stringOne.strmatching()
    print(stringOne.score)
    intOne.int2float_analyze()
    #以上分别计算了图像的相似度、字符串是否被匹配道、以及得到的数字的索引范围和器对应的值，进入综合分析器分析
#     SyntheticAnalyser(pictureOne,stringOne,intOne,HTML,HtmlParser)
#     SyntheticAnalyser.process()
    DependenceTree = DependenceTreeprocess.DependenceTree_fun(HTML,HtmlParser)
    DependenceTree.bracket_split(py_file)
    DependenceTree.output()
    DependenceTree.outword(DependenceTree.print_sent)
    DependenceTree.graphviz_fun()
    DependenceTree.match_compute()
    
    score_computer = SyntheticAnalyser.SCORE_GET(pictureOne,stringOne,intOne,HTML,HtmlParser)
    score_computer.all()
    
    
    pdb.set_trace()
    HTML.end()
    print("finish")


# In[ ]:




