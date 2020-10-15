#coding=utf-8
import os
import sys

Realpath = 'D:\\Documents\\Blog\\posts\\_postsReal\\'
path = 'D:\\Documents\\Blog\\_posts\\'

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
file = open(Realpath+file_name+'.md', 'r', encoding='UTF-8') 
file_context = file.read()
file.close()

str_list = ["@@rp","@@bp","@@pc"]
str_list_real = ["<strong><font color=red>","<strong><font color=blue>","</font></strong>"]
for i in range(len(str_list)):
    #用 "" 替换此字符串中出现的所有
    file_context=file_context.replace(str_list[i],str_list_real[i])

fo = open(path+file_name + "_completed.md", "w", encoding='UTF-8')
fo.write(file_context)
fo.close()


"""
# 添加details元素
"""
# 按行读入文件
file = open(path+file_name+'_completed.md', 'r', encoding='UTF-8') 
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
        temp_str = line[i].lstrip()
        for j in range(0,len(temp_str)):
            if (temp_str[j] == '`'):
                temp=temp+1
            else:
                break
        c = temp
    n = line[i].count('#')
    if (line[i][0:n] != '#'*n):
        temp = 0
        temp_str = line[i].lstrip()
        for j in range(0,len(temp_str)):
            if (temp_str[j] == '#'):
                temp = temp+1
            else:
                break
        n = temp

    if (c == 3):
        code_block = (code_block+1)%2
    if (code_block == 0):
        if (n > 0 and flag[n] == 0):
            line[i] = line[i].lstrip()
            line[i] = str_det[0]+line[i]+str_det[1]
            flag[n] = 1
        elif (n > 0 and flag[n] == 1):
            line[i] = line[i].lstrip()
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
fo = open(path+file_name + "_completed.md", "w", encoding='UTF-8')
fo.writelines(line)
fo.close()

