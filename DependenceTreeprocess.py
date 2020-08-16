#对全文的代码做一次依赖树检测
import htmlprocess
import re
import pdb
from collections import Counter
from graphviz import Digraph
from graphviz import Source
import base64
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class DependenceTree_fun():
    def __init__(self, HTML,HtmlParser):
        self.HTML = HTML
        self.HtmlParser = HtmlParser
        self.left = []
        self.right = []
        self.destination = []
        self.src = []
        self.filename = ""
        self.print_sent = []
        self.print_word = []
        self.left_data =[]
        self.right_data = []
        self.one_hot_des = dict() 
    def line_de_split(self,words):
        regEx = re.compile('\W+')
        word_split = regEx.split(words)
        return word_split
    #搜索括号内部是否还有其他括号
    def function_class(self):
        #对于类和函数内的局部变量，使用全局变量检测
        self.filename = filename
#         regEx = re.compile("here is a decollatorend")
#         word_split = regEx.split( self.HtmlParser.jupyter_data  )
#         pdb.set_trace()
        file = open(self.filename)
        for line in file:
            pass
    def bracket_split(self,filename):
        self.filename = filename
#         regEx = re.compile("here is a decollatorend")
#         word_split = regEx.split( self.HtmlParser.jupyter_data  )
#         pdb.set_trace()
        file = open(self.filename)
        for line in file:
            if line.count("=") > 0 :
                words = line.split("=")
                ll = []
                self.left.append(self.line_de_split(words[0]))
                for ww in words[1:] :
                    ll  += self.line_de_split(ww)
                self.right.append(ll)
            pass # do something
        file.close()
        
        des_word = []
        
        for ll in self.left:
            des_word = des_word + ll
        one_hot_des = dict(Counter( des_word ))
        
        src_word = []
        
        for rr in self.right:
            src_word = src_word + rr
        one_hot_src = dict(Counter( src_word ))
        
        for key,value in one_hot_des.items() :
            self.right_data = []
            counter = 0
            one_index = []
            for ll in self.left :
                if key in ll:
                    one_index.append(self.right[counter])
                counter = counter + 1 
            #同样，在one_hot_src中也必须找到同样的索引
            self.one_hot(one_index)
            self.left_data.append( dict(Counter( self.right_data )).items())
        self.one_hot_des = one_hot_des
        #把 one_hot_des 和  self.left_data 放入 流程图中绘制
        
        #右侧也唯一索引
    def one_hot(self,input_str):
        if isinstance(input_str, (tuple,list) ):
            for ll in range(len(input_str) ):
                self.one_hot(input_str[ll])
        elif isinstance(input_str, str ):
            self.right_data.append(input_str)
    
    def output(self):
        #print 输出结果的长依赖检测
        file = open(self.filename)
        rule_name = r'(.*)print\((.*)\)(.*)'
        compile_name = re.compile(rule_name)
        for line in file:
#             if line.count("format(epoch+1, num_epochs, i+1, total_step, lo") != 0 :
            if compile_name.findall(line) != [] :
                
                self.print_sent.append( compile_name.findall(line) )
                
        #去掉其中的字符串内容，字符分割，在one_hot_src中搜索，并计算全部右依赖
    def outword(self,input_str):
        rule_name2 = r'\'(.*)\''
        compile_name2 = re.compile(rule_name2)
        if isinstance(input_str, (tuple,list) ):
            for ll in range(len(input_str) ):
                self.outword(input_str[ll])
        elif compile_name2.findall(input_str) != []:
            for cc in compile_name2.findall(input_str):
                index = input_str.find(cc, 0)
                input_str = input_str[0:index-1]+input_str[index+len(cc):]
            self.print_word.append( input_str )
        else:
            self.print_word.append( input_str )
    
    def graphviz_fun(self):
        dot = Digraph(comment='The Test Table', format="png")
        dot.attr(rankdir='LR', size='800,900')
        for key,value in self.one_hot_des.items() :
            # 添加圆点A,A的标签是Dot A
            dot.node(key, key)
        counter = 0
        for key,value in self.one_hot_des.items() :
            for key2,value2 in self.left_data[counter]:
                dot.edge(key2, key, str(value2))
            counter = counter + 1

        # 保存source到文件，并提供Graphviz引擎
        dot.save('test-table.gv')  # 保存
        dot.render('output-graph.gv', view=True)
        self.HTML.write_html("<br>")

        self.HTML.create_table()
        self.HTML.start_tr()
        self.HTML.add_td("源码长依赖检测")
        self.HTML.end_tr()
        self.HTML.start_tr()
        f = open("output-graph.gv.png","rb")
        image_base64 = str(base64.b64encode(f.read()), encoding='utf-8')
        self.HTML.add_td("""<img src="data:image/png;base64,""" + image_base64 +  """""/>""")
        self.HTML.end_tr()
        self.HTML.end_table()
#         dot.render('test-table.gv')
        # dot.view()  # 显示
        # 从保存的文件读取并显示
#         s = Source.from_file('test-table.gv')
#         print(s.source)  # 打印代码
        # s.view()  # 显示
#     在多维度空间对以上的值和依赖做聚类分析
    def match_compute(self):
            # 添加圆点A,A的标签是Dot A
        first_dimension = []
        secon_dimension = []
        third_dimension = []
        counter = 0
        for pp in range(pd.DataFrame(self.one_hot_des.items()).shape[0] ):
            for pp2 in range(pd.DataFrame(self.left_data[counter]).shape[0] ):
                first_dimension.append(pp)
                secon_dimension.append(pp2)
                if pd.DataFrame(self.left_data[counter]).loc[pp2][1] > 50 :
                    third_dimension.append(10)
                else:
                    third_dimension.append( pd.DataFrame(self.left_data[counter]).loc[pp2][1] )
            counter = counter + 1
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(first_dimension ,secon_dimension ,third_dimension , c='r', marker='o')
        ax.set_xlabel('dependent variable Label')
        ax.set_ylabel('variable Label')
        ax.set_zlabel('depend num Label')
        plt.savefig("/temp/3d.png")
        plt.show()
        self.HTML.create_table()
        self.HTML.start_tr()
        self.HTML.add_td("三维视图")
        self.HTML.add_td("俯视图")
        self.HTML.add_td("主视图")
        self.HTML.add_td("右视图")
        self.HTML.end_tr()
        self.HTML.start_tr()
        f = open("/temp/3d.png","rb")
        image_base64 = str(base64.b64encode(f.read()), encoding='utf-8')
        self.HTML.add_td("""<img src="data:image/png;base64,""" + image_base64 +  """""/>""")
        
        
        
        
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        plt.xlabel('dependent variable Label')
        plt.ylabel('variable Label')
        ax1.scatter(first_dimension ,secon_dimension,c = 'r',marker = 'o')
        plt.savefig("/temp/2d1.png")
        plt.show()
        
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        plt.xlabel('dependent variable Label')
        plt.ylabel('depend num Label')
        ax1.scatter(first_dimension ,third_dimension,c = 'r',marker = 'o')
        plt.savefig("/temp/2d2.png")
        plt.show()
        
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        plt.xlabel('variable Label')
        plt.ylabel('depend num Label')
        ax1.scatter(secon_dimension,third_dimension,c = 'r',marker = 'o')
        plt.savefig("/temp/2d3.png")
        plt.show()
        
        f = open("/temp/2d1.png","rb")
        image_base64 = str(base64.b64encode(f.read()), encoding='utf-8')
        self.HTML.add_td("""<img src="data:image/png;base64,""" + image_base64 +  """""/>""")
        
        f = open("/temp/2d2.png","rb")
        image_base64 = str(base64.b64encode(f.read()), encoding='utf-8')
        self.HTML.add_td("""<img src="data:image/png;base64,""" + image_base64 +  """""/>""")
        
        f = open("/temp/2d3.png","rb")
        image_base64 = str(base64.b64encode(f.read()), encoding='utf-8')
        self.HTML.add_td("""<img src="data:image/png;base64,""" + image_base64 +  """""/>""")
        
        self.HTML.end_tr()
        self.HTML.end_table()
        
        #使用数据聚类的方法，对三个维度的数据分别聚类分析
        
        
        
        
        
        

            
        
                                                 