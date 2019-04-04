# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 15:03
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : tmodel2.py
# @Software: PyCharm


""" 利用倒排表进行优化 """
import pandas as pd
import matplotlib as mpl
import numpy as np
from nltk.probability import FreqDist
import time

from jiebaSegment import *
from sentenceSimilarity import SentenceSimilarity

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese


def read_corpus():
    qList = []
    # 问题的关键词列表
    qList_kw = []
    aList = []
    data = pd.read_csv('./data/qa_.csv', header=None)
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(t[0])
        qList_kw.append(seg.cut(t[0]))
        aList.append(t[1])
    return qList_kw, qList, aList


def plot_words(wordList):
    fDist = FreqDist(wordList)
    #print(fDist.most_common())
    print("单词总数: ",fDist.N())
    print("不同单词数: ",fDist.B())
    fDist.plot(10)


def invert_idxTable(qList_kw):  # 定一个一个简单的倒排表
    invertTable = {}
    for idx, tmpLst in enumerate(qList_kw):
        for kw in tmpLst:
            if kw in invertTable.keys():
                invertTable[kw].append(idx)
            else:
                invertTable[kw] = [idx]
    return invertTable


def filter_questionByInvertTab(inputQuestionKW, questionList, answerList, invertTable):
    idxLst = []
    questions = []
    answers = []
    for kw in inputQuestionKW:
        if kw in invertTable.keys():
            idxLst.extend(invertTable[kw])
    idxSet = set(idxLst)
    for idx in idxSet:
        questions.append(questionList[idx])
        answers.append(answerList[idx])
    return questions, answers



if __name__ == '__main__':
    # 设置外部词
    seg = Seg()
    seg.load_userdict('./userdict/userdict.txt')
    # 读取数据
    qList_kw, questionList, answerList = read_corpus()

    """简单的倒排索引"""
    # 计算倒排表
    invertTable = invert_idxTable(qList_kw)

    while True:
        question = input("请输入问题(q退出): ")
        time1 = time.time()
        if question == 'q':
            break
        inputQuestionKW = seg.cut(question)
        # 利用关键词匹配得到与原来相似的问题集合
        questionList_s, answerList_s = filter_questionByInvertTab(inputQuestionKW, questionList, answerList,
                                                                  invertTable)
        print(questionList_s)
        if len(questionList_s) > 1:
            questionList = questionList_s
            answerList = answerList_s

        # 初始化模型
        ss = SentenceSimilarity(seg)
        ss.set_sentences(questionList)
        ss.TfidfModel()  # tfidf模型
        # ss.LsiModel()         # lsi模型
        # ss.LdaModel()         # lda模型

        question_k = ss.similarity_k(question, 5)
        print("亲，我们给您找到的答案是： {}".format(answerList[question_k[0][0]]))
        for idx, score in zip(*question_k):
            print("same questions： {},                score： {}".format(questionList[idx], score))
        time2 = time.time()
        cost = time2 - time1
        print('Time cost: {} s'.format(cost))


