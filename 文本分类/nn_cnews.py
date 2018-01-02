#!/usr/bin/env python
# encoding: utf-8


"""
@author:    0604hx
@license:   MIT 
@contact:   zxingming@foxmail.com
@site:      https://github.com/0604hx
@software:  PyCharm Community Edition
@project:   tensorflow-study
@file:      nn_cnews.py
@time:      17/12/30 上午12:40

利用神经网络学习 cnews 预测模型

关于 cnews 详见：http://thuctc.thunlp.org/

本例中使用数据来源：https://github.com/gaussic/text-classification-cnn-rnn

代码参考：https://github.com/dmesquita/understanding_tensorflow_nn

"""
from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr
import jieba
import os

import time

from datetime import timedelta

import sys

DIR = os.path.dirname(__file__)
DATA_TRAIN = os.path.join(DIR, "cnews", "cnews.train.txt")
DATA_TEST = os.path.join(DIR, "cnews", "cnews.test.txt")
DATA_VAL = os.path.join(DIR, "cnews", "cnews.val.txt")
DATA_VOCAL = os.path.join(DIR, "cnews", "vocal.txt")


def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class DataProcessor(object):
    """
    数据预处理

    构建词库：
        对训练数据进行分词后，建立总词库

        1. 读取 训练数据 文件
        2. 逐个对新闻内容进行分词，建立总词库
    """

    def build_vocab(self, data_file, vocab_file, vocab_size=100000):
        start_time = time.time()

        with open_file(data_file) as f:
            counter = Counter()

            i = 1
            for line in f:
                category, content = line.strip().split("\t")
                # 使用 结巴 中文分词
                words = jieba.lcut(content, cut_all=False)
                counter.update([w for w in words if len(w) > 1])
                print("{:<6} {} {}".format(i, category, content))

                i += 1

            print("共 {} 个词，保存到 {}，耗时 {}".format(len(counter), vocab_file, get_time_dif(start_time)))
            pairs = counter.most_common(vocab_size)
            words, _ = list(zip(*pairs))
            open_file(vocab_file, mode='w').write('\n'.join(words) + '\n')

    def read_vocab(self, vocab_file):
        """
        读取词表
        :param vocab_file:
        :return:
        """
        words = open_file(vocab_file).read().strip().split("\n")
        word_id = dict(zip(words, range(len(words))))

        return words, word_id

    def categories(self):
        categories = ['体育', '财经', '房产', '家居',
                      '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        cat_to_id = dict(zip(categories, range(len(categories))))
        return categories, cat_to_id

    def read_data(self, filename, word_to_id, cat_to_id, max_length=200):
        """
        读取数据文件，返回 数据、结果 的集合

        数据格式：[数据条数, max_length] 张量
        结果格式：[数据条数, 10] one-hot 格式的分类结果
        :param filename:
        :param word_to_id:
        :param cat_to_id:
        :param max_length:  数据集的长度，默认 200
        :return:
        """
        start_time = time.time()

        x_data = []
        y_data = []

        print("开始读取数据 ：%s" % filename)
        with open_file(filename) as f:
            i = 1
            for line in f:
                category, content = line.strip().split("\t")
                # 使用 结巴 中文分词
                origin_words = jieba.lcut(content, cut_all=False)
                words = [w for w in origin_words if len(w) > 1]
                print(words)

                data = np.zeros(max_length, dtype=float)

                for ii in range(min(max_length, len(words))):
                    data[ii] = word_to_id[words[ii]] if words[ii] in word_to_id else 0

                x_data.append(data)

                category_ = np.zeros(10, dtype=float)
                category_[cat_to_id[category]] = 1

                y_data.append(category_)

                i += 1
                if i > 1:
                    break

        print("{} 条数据读取完成，耗时 {}，来源 {}".format(len(x_data), get_time_dif(start_time), filename))
        return x_data, y_data


def train(train_file):
    """
    训练数据
    :return:
    """
    print("训练数据：", train_file)
    x_data, y_data = processor.read_data(DATA_TRAIN, word_ids, categories_id)
    print(x_data[0])
    print(y_data[0])


if __name__ == '__main__':
    processor = DataProcessor()
    if not os.path.exists(DATA_VOCAL):
        print("检测到词表未生成，即将创建...")
        processor.build_vocab(DATA_TRAIN, DATA_VOCAL)

    words, word_ids = processor.read_vocab(DATA_VOCAL)
    categories, categories_id = processor.categories()
    print("分类", categories_id)

    if sys.argv[1] == 'train':
        train(DATA_TRAIN)
