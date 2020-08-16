import os.path
from html.parser import HTMLParser
import requests
import os
import shutil
import urllib.request
from urllib import request
import pdb 
import re
class myHTMLParser(HTMLParser):     # 创建HTML解析类
    def __init__(self):
        HTMLParser.__init__(self)
        self.gifs_urls = []         # 创建列表，保存gif
        self.jpgs_urls = []         # 创建列表，保存jpg
        self.num = 0
        self.data = ''
        self.flag = False
        self.jupyter_data = ""
        self.jupyter_cell_tags = [""]
        self.jupyter_counter = 0
        self.jupyter_bracket_start = []
        self.jupyter_bracket_end = []
        self.bracket_flag = 0
        self.tags = ""
        self.jupyter_word = ""
    # 重写HTMLParser中的内置方法
    def handle_starttag(self, tags, attrs):  # 处理起始标记
        if tags == 'img':   # 处理图片
            self.tags = "img"
            for attr in attrs:
                #print(attr[0])
                if 'png' in attr[1]:
                    self.gifs_urls.append(attr)    # 添加到gif列表
                elif 'jpg' in attrs and 'https' in attrs:
                    self.jpgs_urls.append(attr)    # 添加到jpg列表
                else:
                    pass
        if tags == 'div':
            self.tags = "div"
            for k,v in attrs:#遍历div的所有属性以及其值
                if k == 'class' and v == "output_subarea output_stream output_stdout output_text":#确定进入了<div class='cell'>
                    #self.index = self.index + 1
                    self.flag = True
                    self.data = self.data + '<'
                    return
                if k == 'class' and v == "output_text output_subarea output_execute_result":
                    self.flag = True
                    self.data = self.data + '<'
                    return
#         if tags == 'span':
#             self.tags = "span"
#             for k,v in attrs:#遍历div的所有属性以及其值
#                 self.flag = True

    #覆盖endtag方法
    def handle_endtag(self, tags):
        #pass     
        #print("遇到结束标签:{} 开始处理:{}".format(tag, tag))
        if(self.flag == True and (self.tags == 'img' or self.tags == 'div')):
            if(self.data[-2:]=="\n"):
                self.data = self.data + '>\n'
            else:
                self.data = self.data + '\n>\n'
            self.flag = False
            #遇到tr结束,增加一个回车
#         if self.flag == True and self.tags == "span":
#             print(self.jupyter_word)
#             if( self.jupyter_word.count('(') > 0 ):#确定进入了<div class='cell'>  self.data
#                 self.jupyter_bracket_start.append(self.jupyter_counter)
# #                     括号开加一，括号闭减一,等于0是一个完备的括号
#                 if self.bracket_flag == 0 :
#                     self.jupyter_data = self.jupyter_data + "here is a decollatorstart"
#                 self.bracket_flag = self.bracket_flag + self.jupyter_word.count('(')

#             if( self.jupyter_word.count(')') > 0 ):
#                 self.jupyter_bracket_end.append(self.jupyter_counter)
#                 self.bracket_flag = self.bracket_flag - self.jupyter_word.count(')')
#                 if self.bracket_flag == 0 :
#                     self.jupyter_data = self.jupyter_data + "here is a decollatorend"
#             print(self.bracket_flag)
#             self.jupyter_counter = self.jupyter_counter + 1
#             self.flag = False
    # 自定义的方法
    def get_pngs(self):     # 返回gif列表
        return self.gifs_urls

    def get_jpgs(self):     # 返回jpg列表
        return self.jpgs_urls
    '''
    # 自定义的方法，获取页面
    def getHTML(self,url):
        req=request.Request(url,method='GET')
        html=request.urlopen(req,timeout=30)
        return html.read()
    '''
    # 自定义的方法，批量下载图片
    def downImgs(self,img_urls,n=10,path='./'):
        count=0
        for url in img_urls:
            #print(url)
            request.urlretrieve(url=url[1],filename='/temp/{0}{1}{2}'.format(path,count,'.png'))
            count=count+1
#         print('共检测到%d张图片' %(count))
        self.num = count
        return(count)
            
    def read_file(self,filename):
        #pdb.set_trace()
        fp = open(filename,'r',encoding='utf-8')
        content = fp.read()
        fp.close()
        return content
    def handle_data(self, data):
        #pass
        #print("遇到数据:{} 开始处理:{}".format(data, data))
        if(self.flag == True and (self.tags == 'img' or self.tags == 'div')):
            #pdb.set_trace()
            #data = data.replace('\n','')#替换字段中的回车
            #data = data.replace('  ','')#替换字段中的连续两个空格
            self.data = self.data + data
#         if(self.flag == True and self.tags == 'span'):
#             #pdb.set_trace()
#             #data = data.replace('\n','')#替换字段中的回车
#             #data = data.replace('  ','')#替换字段中的连续两个空格
#             self.jupyter_data = self.jupyter_data + " " + data
#             self.jupyter_word = data

    def write_file(self,filename,content):
        fp = open(filename,'a+',encoding='utf-8')
        fp.write(content)
        fp.close()
    def split_str(self,data):
        return data.split("\n")

class html_display():     # 创建HTML解析类
    def __init__(self,html_reportfile):
        self.gifs_urls = []         # 创建列表，保存gif
        self.jpgs_urls = []         # 创建列表，保存jpg
        self.num = 0
        self.fd = 0
        self.file = html_reportfile
    def write_html(self,message):
        self.fd.write(message)
        return 0
    def create_table(self):
        self.write_html("""<table class="default-table" width="400" height="400" border="1">""")
    def end_table(self):
        self.write_html("</table>")
    def start_tr(self):
        self.write_html(""" <tr VALIGN="TOP" align=left > """)
    def end_tr(self):
        self.write_html(""" </tr> """)
    def add_td(self,str_out_html):
        back = "<td>"+str_out_html+"</td>"
        self.write_html(back)
    def style(self,str_out_html,color):
        back = """<font size="3" color= """ + color + " > " +  str_out_html + "</font>"
        self.write_html(back)
    def writr_picture(self,image_np):
        image = cv2.imencode('.png', image_np)[1]
        image_code = str(base64.b64encode(image))[2:-1]
        data='data:image/png;base64,' + image_code
        mss = """<div class="output_png output_subarea "> <img src= """
        #pdb.set_trace()
        self.fd.write(mss)
        self.fd.write(data)
        self.fd.write(">  </div>")
        #pdb.set_trace()
    def start(self):
#         try:
#             self.fd = open(self.file,'w')
#         except IOError:
#             #找不到文件时提示文件不存在
#             print("not exist html file")
        file_object = open(self.file, 'w+')
        self.fd = file_object
        message_1 = """
            <html>
            <head> 
            <meta charset="utf-8" /> 
            <title>虚拟仿真平台实验成绩单</title> 

            <style>

            table.default-table{  
                /* -moz-border-radius: 5px;  
                -webkit-border-radius:5px;  
                border-radius:5px; */  
                width: 100%;  
                border:solid #333;   
                border-width:1px 0px 0px 1px;  
                font-size: #333;  
                border-collapse: collapse;  
                border-spacing: 0;  
            }  
            table.default-table tbody tr{  
                height: 20px;  
                line-height: 20px;  
            }  
            table.default-table tbody tr.odd{  
                background-color: #fff;  
            }  
            table.default-table tbody tr.even{  
                background-color: #F5F5F5;  
            }  
            table.default-table tbody tr:hover{  
                background-color: #eee;  
            }  
            table.default-table tbody tr th,table.default-table tbody tr td{  
                padding:3px 5px;  
                text-align: left;  
                /* border: 1px solid #ddd; */  
                border:solid #333;   
                border-width:0px 1px 1px 0px;   
            }  
            table.default-table tbody tr th{  
                font-weight: normal;  
                text-align: center;  
            }  

            table.default-table tbody tr td.tac{  
                text-align: center;  
            }  
            table.default-table tbody tr td a:hover{  
                color:#0080c0;  
            }   
            </style> 

            <style> 
            .divcss5{ font-family:"宋体";font-weight: bold;}
            </style> 

            <style> 
            .divcss6{ font-family:"宋体";font-size:20px;position: absolute;left: 50%;font-weight: bold;}
            </style> 

            </head> 
            <body>
            """
        self.write_html(message_1)
#         pdb.set_trace()
    def end(self):
        self.write_html("""</body></html>""")
        self.fd.close()
        return 0