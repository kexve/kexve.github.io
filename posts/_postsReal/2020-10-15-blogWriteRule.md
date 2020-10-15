---  
layout: post  
title: 博客书写规范
categories: Blog  
---  

## 书写规范
1. 凡是标题, 顶格写  
2. 除图片行, 标题行不空格, 其余行末尾均空两格  
3. 特殊语法用`@@ + 'ab'`简写, 如红笔`+ rp`, 蓝笔`+ bp`, 笔结束`+ pc`  
4. 最多有六级标题, 一般不用一级标题, h1-h6
5. 引用块, 顶格写`> `, 每行都加, 整个引用块结束, 加一个空行, 块内无空行
6. 代码头, 顶格写
7. 列表有多行在一个`li`下, 前面空4格

## 测试区域
![Image](https://s3.jpg.cm/2020/10/15/thPvw.png)

### 标题
 #### 这是标题吗
这里是内容
#### 这是标题吗#
这里是内容
#### 标题可以加空格吗    
这里是内容

### 引用块
> 这是引用块  
> 一起  

> 这也一起吗

> +1

### 代码区域
 ``` txt
这是代码区域吗, 没顶格
#### 这是标题吗
这个呢####
 ```
### 图片
![Image](https://s3.jpg.cm/2020/10/15/thPvw.png)        
![Image](https://s3.jpg.cm/2020/10/15/thPvw.png)
![Image](https://s3.jpg.cm/2020/10/15/thPvw.png)

## Python代码

``` python
#coding=utf-8
import os
import sys

"""
# 得到file_name
"""
if(len(sys.argv) == 1):
    file_name = input("input file name (no suffix): ") # 将所有看成字符串
else:
    file_name = sys.argv[1] # cmd直接传参

file_name = file_name.replace('.md','')


"""
# 将@@bp和@@rp和@@pc变为相应的值
"""
# 整个文件读入
file = open(file_name+'.md', 'r', encoding='UTF-8') 
file_context = file.read()
file.close()

str_list = ["@@rp","@@bp","@@pc"]
str_list_real = ["<font color=red>","<font color=blue>","</font>"]
for i in range(len(str_list)):
    #用 "" 替换此字符串中出现的所有
    file_context=file_context.replace(str_list[i],str_list_real[i])

fo = open("../../_posts/"+file_name + "_completed.md", "w", encoding='UTF-8')
fo.write(file_context)
fo.close()


"""
# 添加details元素
"""
# 按行读入文件
file = open("../../_posts/"+file_name+'_completed.md', 'r', encoding='UTF-8') 
line = file.readlines()
line.append("\n")
file.close()

flag = [0,0,0,0,0,0,0]
code_block = 0
str_det = ["@@dt\n","@@ds\n","@@dc\n"]
for i in range(len(line)):
    c = line[i].count('`')
    if (line[i][0:c] != '`'*c):
        temp = 0
	str_temp = line[i].lstrip()
        for j in range(0,len(str_temp)):
            if (str_temp[j] == '`'):
                temp=temp+1
            else:
                break
        c = temp
    n = line[i].count('#')
    if (line[i][0:n-1] != '#'*n):
        temp = 0
	str_temp = line[i].lstrip()
        for j in range(0,str_temp)):
            if (str_temp[j] == '#'):
                temp = temp+1
            else:
                break
        n = temp

    if (c == 3):
        code_block = (code_block+1)%2
    if (code_block == 0):
        if (n > 0 and flag[n] == 0):
            line[i] = str_det[0]+line[i]+str_det[1]
            flag[n] = 1
        elif (n > 0 and flag[n] == 1):
            line[i] = str_det[2]*sum(flag[n:5])+str_det[0]+line[i]+str_det[1]
            for j in range(n+1,6):
                flag[j] = 0
        elif (i == len(line)-1):
            line[i] = line[i]+str_det[2]*sum(flag)
        elif (n == 0):
            line[i] = line[i].rstrip()+'  '+'\n'

# 删除空白行
j = 0 
for i in range(len(line)):
    if (line[j].isspace()):
        line.pop(j)
    else:
        j += 1
quote_block = 0
for i in range(len(line)):
    if (line[i][0:2] == '> ' and quote_block == 0):
        quote_block = 1
    elif (line[i][0:2] != '> ' and quote_block == 1):
        line[i] = '\n'+line[i]
        quote_block = 0
    if (line[i][0:2] == '!['):
        line[i] = line[i].rstrip()+'\n'

# 输出到.md文件
fo = open("../../_posts/"+file_name + "_completed.md", "w", encoding='UTF-8')
fo.writelines(line)
fo.close()
```
