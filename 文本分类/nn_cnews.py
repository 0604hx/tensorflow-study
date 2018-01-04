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
import json
import tensorflow as tf
from collections import Counter
import numpy as np
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

SAVE_DIR = "model/cnews"


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

        json_file = os.path.join(os.path.dirname(filename), os.path.basename(filename) + ".done.json")

        if not os.path.exists(json_file):
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

                    data = np.zeros(max_length, dtype=float)

                    for ii in range(min(max_length, len(words))):
                        data[ii] = word_to_id[words[ii]] if words[ii] in word_to_id else 0

                    x_data.append(data.tolist())

                    category_ = np.zeros(10, dtype=float)
                    category_[cat_to_id[category]] = 1

                    y_data.append(category_.tolist())

                    i += 1

                # 保存
                open_file(json_file, 'w').write(json.dumps([x_data, y_data]))
        else:
            print("检测到 %s 存在，直接读取[length=%d]..." % (json_file, max_length))
            with open_file(json_file) as f:
                ds = json.loads(f.read())
                x_data = ds[0]
                y_data = ds[1]

        print("{} 条数据读取完成，耗时 {}，来源 {}".format(len(x_data), get_time_dif(start_time), filename))
        return x_data, y_data


def train_multilayer_perception(input_tensor, weights, biases):
    """
    构建 3 层神经网络

    :param input_tensor:
    :param weights:
    :param biases:
    :return:
    """
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)

    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)

    # Output layer
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

    return out_layer_addition


def train(train_file):
    """
    训练数据

    2018年1月4日
        rate = 0.1  epochs = 25 length = 200 ，loss = 2.3
        rate = 0.1  epochs = 25 length = 300 ，loss = 2.4
        rate = 0.01  epochs = 25 length = 200 ，loss = 50.3
        rate = 0.2  epochs = 25 length = 200 ，loss = 3.2 （只需 5 轮训练）
        rate = 0.15  epochs = 25 length = 200 ，loss = 2.8 （10轮训练）
        rate = 0.17  epochs = 25 length = 200 ，loss = 2.7 （10轮训练）
        rate = 0.12  epochs = 25 length = 200 ，loss = 2.4 （12轮训练）

    :return:
    """
    print("训练数据：", train_file)

    # Parameters
    learning_rate = 0.2
    training_epochs = 10
    batch_size = 150
    display_step = 1

    # Network Parameters
    n_hidden_1 = 100  # 1st layer number of features
    n_hidden_2 = 50  # 2nd layer number of features
    n_input = 200  # Words in vocab
    n_classes = len(categories_id)  # Categories: graphics, sci.space and baseball

    # 加载数据
    x_data, y_data = processor.read_data(DATA_TRAIN, word_ids, categories_id, n_input)
    print("训练数据 {} 条， shape={}".format(len(x_data), len(x_data[0])))

    input_tensor = tf.placeholder(tf.float32, [None, n_input], name="input")
    output_tensor = tf.placeholder(tf.float32, [None, n_classes], name="output")

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    prediction = train_multilayer_perception(input_tensor, weights, biases)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = int(len(x_data) / batch_size)

        for epoch in range(training_epochs):
            avg_cost = 0.
            for i in range(total_batch):
                offset = i * batch_size
                x_batch = x_data[offset: offset + batch_size]
                y_batch = y_data[offset: offset + batch_size]

                c, _ = sess.run([loss, optimizer], feed_dict={input_tensor: x_batch, output_tensor: y_batch})
                avg_cost += c / total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # 配置 Saver
        saver = tf.train.Saver()
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        saver.save(sess, save_path=os.path.join(SAVE_DIR, "cnews"))
        print("模型保存到 %s" % SAVE_DIR)

        # # 进行预测
        x_test, y_test = processor.read_data(DATA_TEST, word_ids, categories_id, n_input)
        # y_pred = sess.run(prediction, feed_dict={input_tensor: x_test})
        # for i in range(5):
        #     print(y_pred[i])
        #     print(y_test[i])

        # Test model
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        batch_x_test = x_test[0: batch_size]
        batch_y_test = y_test[0: batch_size]
        print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))


def test(test_file):
    """

    :param test_file:
    :return:
    """
    # 加载模型
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=os.path.join(SAVE_DIR, "cnews"))

    # print(prediction)


if __name__ == '__main__':
    processor = DataProcessor()
    if not os.path.exists(DATA_VOCAL):
        print("检测到词表未生成，即将创建...")
        processor.build_vocab(DATA_TRAIN, DATA_VOCAL)

    words, word_ids = processor.read_vocab(DATA_VOCAL)
    categories, categories_id = processor.categories()
    print("分类", categories_id)

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train(DATA_TRAIN)
