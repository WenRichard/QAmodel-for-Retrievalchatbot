# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 16:55
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : analysis.py
# @Software: PyCharm
import json
import nltk
import numpy as np
import scipy.spatial.distance as distance
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt

from tmodel import clean_words
'''Part 2 理解数据（可视化分析/统计信息'''
# TODO: 统计一下在qlist 总共出现了多少个单词？ 总共出现了多少个不同的单词？需要做简单的分词，对于英文我们根据空格来分词即可，
# TODO： 从上面的图中能观察到什么样的现象？ 这样的一个图的形状跟一个非常著名的函数形状很类似，能所出此定理吗？

#通过观察词频的分布类似有zip's law, 是一个很著名的现象，包括在社交网络里面（比如大V的好友数非常多，其他人的好友数指数级下降）
# TODO: 在qlist和alist里出现次数最多的TOP 10单词分别是什么？
#出现最多的十个单词见图中单词
# TODO: 统计一下qlist中每个单词出现的频率，并把这些频率排一下序，然后画成plot. 比如总共出现了总共7个不同单词，而且每个单词出现的频率为 4, 5,10,2, 1, 1,1
#       把频率排序之后就可以得到(从大到小) 10, 5, 4, 2, 1, 1, 1. 然后把这7个数plot即可（从大到小）
#       需要使用matplotlib里的plot函数。y轴是词频


def plot_words(wordList):
    fDist = FreqDist(wordList)
    #print(fDist.most_common())
    print("单词总数: ",fDist.N())
    print("不同单词数: ",fDist.B())
    fDist.plot(10)
all_question = " ".join(questionList)
qWordLst = clean_words(all_question)
plot_words(qWordLst)
