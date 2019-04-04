# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 9:21
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : sentenceSimilarity.py
# @Software: PyCharm


import gc
import tqdm
import numpy as np
from gensim import corpora, models, similarities
from sentence import Sentence
from collections import defaultdict


class SentenceSimilarity():

    def __init__(self, seg):
        self.seg = seg

    def set_sentences(self, sentences):
        self.sentences = []
        for i in range(0, len(sentences)):
            self.sentences.append(Sentence(sentences[i], self.seg, i))

    # 获取切过词的句子
    def get_cuted_sentences(self):
        cuted_sentences = []

        for sentence in self.sentences:
            cuted_sentences.append(sentence.get_cuted_sentence())

        return cuted_sentences

    # 构建其他复杂模型前需要的简单模型
    def simple_model(self, min_frequency = 1):
        self.texts = self.get_cuted_sentences()

        # 删除低频词
        frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                frequency[token] += 1
        self.texts = [[token for token in text if frequency[token] > min_frequency] for text in self.texts]
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]

    # tfidf模型
    def TfidfModel(self):
        self.simple_model()

        # 转换模型
        self.model = models.TfidfModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]

        # 创建相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    # lsi模型
    def LsiModel(self):
        self.simple_model()

        # 转换模型
        self.model = models.LsiModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]

        # 创建相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    # lda模型
    def LdaModel(self):
        self.simple_model()

        # 转换模型
        self.model = models.LdaModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]

        # 创建相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    # 对新输入的句子（比较的句子）进行预处理
    def sentence2vec(self, sentence):
        sentence = Sentence(sentence, self.seg)
        vec_bow = self.dictionary.doc2bow(sentence.get_cuted_sentence())
        return self.model[vec_bow]

    def bow2vec(self):
        vec = []
        length = max(self.dictionary) + 1
        for content in self.corpus:
            sentence_vectors = np.zeros(length)
            for co in content:
                sentence_vectors[co[0]] = co[1]  # 将句子出现的单词的tf-idf表示放入矩阵中
            vec.append(sentence_vectors)
        return vec

    # 求最相似的句子
    # input: test sentence
    def similarity(self, sentence):
        sentence_vec = self.sentence2vec(sentence)

        sims = self.index[sentence_vec]
        sim = max(enumerate(sims), key=lambda item: item[1])

        index = sim[0]
        score = sim[1]
        sentence = self.sentences[index]

        sentence.set_score(score)
        return sentence  # 返回一个类

        # 求最相似的句子
    def similarity_k(self, sentence, k):
        sentence_vec = self.sentence2vec(sentence)

        sims = self.index[sentence_vec]
        sim_k = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)[:k]

        indexs = [i[0] for i in sim_k]
        scores = [i[1] for i in sim_k]
        return indexs, scores
