# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 9:20
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : sentence.py
# @Software: PyCharm


from jiebaSegment import Seg


class Sentence(object):

    def __init__(self, sentence, seg, id=0):
        self.id = id
        self.origin_sentence = sentence
        self.cuted_sentence = self.cut(seg)

    # 对句子分词
    def cut(self, seg):
        return seg.cut_for_search(self.origin_sentence)

    # 获取切词后的词列表
    def get_cuted_sentence(self):
        return self.cuted_sentence

    # 获取原句子
    def get_origin_sentence(self):
        return self.origin_sentence

    # 设置该句子得分
    def set_score(self, score):
        self.score = score
