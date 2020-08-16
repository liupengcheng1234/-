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
class SCORE_GET:
    def __init__(self,pictureOne,stringOne,intOne,HTML,HtmlParser):
        self.pictureOne = pictureOne
        self.stringOne = stringOne
        self.intOne = intOne
        self.HTML = HTML
        self.HtmlParser = HtmlParser
        
    
    #计分使用多维度的最小二乘算法来计算
    def all(self):
        epoch_score = self.picture_get_point()
        string_score = self.string_point()
        float_score = self.float_int_point()
        
        
        total= sum ( np.array(self.pictureOne.scorevalue).astype(float).T *epoch_score) + \
            sum( np.array(self.stringOne.scorevalue).astype(float).T * string_score) + \
            sum(np.array(self.intOne.scorevalue).astype(float).T * float_score )
        
        score = total /( sum( np.array(self.pictureOne.scorevalue).astype(float) ) + sum(np.array(self.stringOne.scorevalue).astype(float)) + sum(np.array(self.intOne.scorevalue).astype(float) ) )
        
        print(score)
        return score 
        
            
    def picture_get_point(self):
        #预处理
        min_get = np.array([self.pictureOne.min_value,self.pictureOne.max_value]).min(0)
        max_get = np.array([self.pictureOne.min_value,self.pictureOne.max_value]).max(0)

        set1 = list(set(self.pictureOne.degree))
        set2 = sorted(set1)
        #计算可用总分值
        total_score = 100
        for j in set2:
            #pdb.set_trace()
            #按照等级来划分最大的成绩，例如没有属性1最大的得分将为80分，而不是100分
            if(self.pictureOne.degree.index(j) != []):
                total_score =  self.pictureOne.degree_score[ self.pictureOne.degree.index(j) ]
                break
            else:
                print(error)

        #其次检测得分的范围，如果超出了定义的范围，将失去这一项目的得分，并且计为0分
        epoch_score = np.zeros(  [self.pictureOne.attr_value.shape[0]]    )
        for m in range(self.pictureOne.attr_value.shape[0]):
            a_logic = operator.ge(max_get,self.pictureOne.attr_value[m,:])
            b_logic = operator.ge(self.pictureOne.attr_value[m,:],min_get)
            #pdb.set_trace()
            if( a_logic.all() and b_logic.all() ):
                #计算各项的分值

                pass     
            else:
                #pdb.set_trace()
                a_index = np.argwhere(a_logic == False)
                b_index = np.argwhere(b_logic == False)
                if( len(a_index) > 0 ):
                    print(a_index)
                    #pdb.set_trace()
                    for kk2 in a_index:
                        kk1 = m
                        kk2 = kk2[0]
                        attr_value[kk1,kk2] = min_value[kk2]
                        #pdb.set_trace()
                        print("第%d张图片的第 %d 个属性超出范围"%( (kk1+1),(kk2+1) ) )

                if( len(b_index) > 0 ):
                    print(b_index)
                    for kk2 in b_index:
                        kk1 = m
                        kk2 = kk2[0]
                        attr_value[kk1,kk2] = min_value[kk2]
                        print("第%d张图片的第 %d 个属性超出范围"%(  (kk1+1),(kk2+1) ) )

                #print(attr_value.index(False))
                pass
            attr_value_seco = ( abs((np.array(self.pictureOne.attr_value[m,:])- np.array(self.pictureOne.min_value ) ) ) / abs( np.array(self.pictureOne.max_value)-np.array(self.pictureOne.min_value)) ) * total_score
            print("第%d张图片的各项成绩为:"%(m+1))
            print(attr_value_seco)
            weights_2 = np.array(self.pictureOne.weights) / sum(self.pictureOne.weights)
            epoch_score[m] = sum(attr_value_seco.T * weights_2)
        return epoch_score

        #pdb.set_trace()
#         print(epoch_score)
        return (sum(epoch_score) / self.pictureOne.attr_value.shape[0])
    def string_point(self):
        print( self.stringOne.score )
        return self.stringOne.score
    def float_int_point(self):
        print ( self.intOne.value  )
        count = 0
        int_score = []
        for ii in self.intOne.minrange:
            if( isinstance(   self.intOne.minvalue[count]    ,list) ):
                min_get = np.array([self.intOne.minvalue[count],self.intOne.maxvalue[count]]).min(0)
                max_get = np.array([self.intOne.minvalue[count],self.intOne.maxvalue[count]]).max(0)
                int_score.append( abs((np.array(self.intOne.value[count][0])- np.array( min_get ) ) ) / abs( np.array(max_get)-np.array(min_get)) )
            else:
                try:
                    isinstance(   self.intOne.minvalue[count],str)
                    isinstance(  self.intOne.value[count][0][0] ,str)
                    float(self.intOne.minvalue[count])
                    float(self.intOne.value[count][0][0])
                    min_get = min([self.intOne.minvalue[count],self.intOne.maxvalue[count]])
                    max_get = min([self.intOne.minvalue[count],self.intOne.maxvalue[count]])
                    int_score.append( abs((np.array( float(self.intOne.value[count][0][0]))- np.array( min_get ) ) ) / abs( np.array(max_get)-np.array(min_get)) )
                except:
                    int_score.append( 30 )
            count = count + 1
        return int_score
    def other_point(self):
        pass
            
        