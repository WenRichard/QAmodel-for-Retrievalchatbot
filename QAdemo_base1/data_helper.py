# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 17:06
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : data_helper.py
# @Software: PyCharm

import pandas as pd
import numpy as np
file1 = './data/raw/train.txt'
file2 = './data/raw/test.txt'
file3 = './data/raw/qa.txt'

dic1 = {}
q_all = []
q_file = [file1, file2]
for i in q_file:
    with open(i, 'r') as f1:
        for r1 in f1:
            id, q1, q2, label = r1.split("|")
            q_all.append(''.join(q1.split(' ')))
            q_all.append(''.join(q2.split(' ')))
print(q_all)
print('总共有{}个问题'.format(len(q_all)))
print('总共有{}个不重复的问题'.format(len(set(q_all))))
for i, j in enumerate(set(q_all)):
    dic1[j] = i+1
print(dic1)


dic2 = {}
all = []
with open(file3, 'r', encoding='utf-8') as f1:
    for r1 in f1:
        id, q, a = r1.split("|")
        part = [''.join(q.split(' ')).strip(), ''.join(a.split(' ')).strip()]
        all.append(part)

df = pd.DataFrame(all, columns = ['q', 'a'])
sin = df.drop_duplicates(subset='q', keep='first')
print(sin)
sin.to_csv('./data/qa_.csv', encoding='utf-8', index=0, header=0)
true = sin.q.tolist()
print(true)

tr = []
with open(file1, 'r') as f1:
    for r1 in f1:
        id, q1, q2, label = r1.split("|")
        q_all.append(''.join(q1.split(' ')))
        q_all.append(''.join(q2.split(' ')))
        if ''.join(q1.split(' ')).strip() or ''.join(q2.split(' ')).strip() in true:
            pa = [id, ''.join(q1.split(' ')).strip(), ''.join(q2.split(' ')).strip(), label.strip()]
            print(pa)
            tr.append(pa)
print(len(tr))
