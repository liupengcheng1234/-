import sys    
#导入urllib.request库
import os.path
from html.parser import HTMLParser
import numpy as np
import xml.sax
import imageprocess
import htmlprocess

class xmlHandler( xml.sax.ContentHandler ):
    def __init__(self,pictureOne,stringOne,intOne):
        self.maxvalue=""
        self.minvalue=""
        self.group=""
        self.value=""
        self.scorevalue=""
        self.type=""
        self.CurrentData = ""
        self.pictureOne = pictureOne
        self.stringOne = stringOne
        self.intOne = intOne
        self.counter = 0
    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "body":
            print("*****body*****")

    # 元素结束事件处理
    def endElement(self, tag):
        if tag == "body":
            print("*****end end*****")
            #调用处理函数去执行分类，正对不同的标签分别评分
            if self.type=="int":
                self.intOne.group.append(self.group)
                self.intOne.scorevalue.append(self.scorevalue)
                self.intOne.maxvalue.append(self.maxvalue)
                self.intOne.minvalue.append(self.minvalue)
            if self.type=="str":
                self.stringOne.group.append(self.group)
                self.stringOne.value.append(self.value)
                self.stringOne.scorevalue.append(self.scorevalue)
            if self.type=="image":
                self.pictureOne.group.append(self.group)
                self.pictureOne.value.append(self.value)
                self.pictureOne.scorevalue.append(self.scorevalue)
            self.intOne.typelist.append(self.type)
            self.intOne.typeaddr.append(self.counter)
            self.counter = self.counter + 1
        elif self.CurrentData == "maxvalue":
            print("maxvalue"+self.maxvalue)
        elif self.CurrentData == "minvalue":
            print("minvalue"+self.minvalue)
        elif self.CurrentData == "group":
            print("group"+self.group)
        elif self.CurrentData == "value":
            print("value"+self.value)
        elif self.CurrentData == "scorevalue":
            print("value"+self.scorevalue)
        elif self.CurrentData == "type":
            print("type"+self.type)
        else:
            pass
        self.CurrentData = ""

    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "maxvalue":
            self.maxvalue  = content
        elif self.CurrentData == "minvalue":
            self.minvalue = content
        elif self.CurrentData == "group":
            self.group = content
        elif self.CurrentData == "value":
            self.value = content
        elif self.CurrentData == "scorevalue":
            self.scorevalue = content
        elif self.CurrentData == "type":
            self.type = content

