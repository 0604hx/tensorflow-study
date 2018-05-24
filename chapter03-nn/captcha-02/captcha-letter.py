# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       captcha-letter.py
@time       2018/5/24 14:54

2018年5月24日
    AdamOptimizer , 0.001, prob=1.
    10 轮 准确率达到 0 %
    20 轮 准确率达到 1.0 %
    45 轮 准确率达到 8.9 %
    55 轮 准确率达到 19.8%
    65 轮 准确率达到 32.10 %
    75 轮 准确率达到 42.90 %
    85 轮 准确率达到 50.10 %
    95 轮 准确率达到 65.60 %
    100轮 准确率达到 64.40 %

    RMSPropOptimizer, 0.001, prob=1.
    10 轮 准确率达到 11.50 %
    20 轮 准确率达到 47.00 %
    30 轮 准确率达到 52.50 %
    45 轮 准确率达到 80.70 %
    55 轮 准确率达到 85.40 %
    65 轮 准确率达到 86.70 %
    75 轮 准确率达到 88.40 %
    85 轮 准确率达到 90.20 %
    95 轮 准确率达到 89.90 %
    100轮 准确率达到 91.50 %

    MSPropOptimizer, 0.003, prob=1. | 0.8, loss=tf.reduce_mean
    10 轮 准确率达到 46.70 %      2.50 %
    20 轮 准确率达到 75.50 %      10.20 %
    30 轮 准确率达到 79.50 %      14.90 %
    45 轮 准确率达到 90.90 %      18.30 %
    55 轮 准确率达到 92.00 %      20.40 %
    65 轮 准确率达到 91.90 %      20.70 %
    75 轮 准确率达到 94.10 %      22.50 %
    85 轮 准确率达到 94.10 %      21.90 %
    95 轮 准确率达到 93.40 %      20.50 %
    100轮 准确率达到 93.90 %      21.50 %

    MSPropOptimizer, 0.005, prob=1. | 0.8, loss=tf.reduce_mean
    10 轮 准确率达到 55.40 %      07.80 %
    20 轮 准确率达到 75.30 %      18.60 %
    30 轮 准确率达到 87.60 %      20.40 %
    45 轮 准确率达到 92.60 %      20.20 %
    55 轮 准确率达到 93.40 %      19.20 %
    65 轮 准确率达到 92.60 %      21.90 %
    75 轮 准确率达到 94.50 %      21.90 %
    85 轮 准确率达到 94.70 %      24.30 %
    95 轮 准确率达到 94.30 %      21.80 %
    100轮 准确率达到 94.40 %      22.50 %

    学习率、 Keep_prob 影响很大
"""

import os
from time import clock

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from color import C


class Config(object):
    NAME = "纯字母（大小写）验证码模型"
    VERSION = "1.1"

    CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"  # 验证码中的字符
    CHAR_LEN = len(CHARS)
    TEXT_LEN = 6

    DIR_TRAIN = "F:/data/captcha/letter-1w"
    DIR_TEST = "F:/data/captcha/letter-test"
    DIR_MODEL = "./models-letter"
    SUFFIX = ".jpg"

    CAPTCHA_LEN = 6
    CAPTCHA_WIDTH = CAPTCHA_LEN * 15
    CAPTCHA_HEIGHT = 24
    CAPTCHA_TUNNEL = 3

    NN_KEEP_PROB = 1.
    NN_IN_SIZE = CAPTCHA_HEIGHT * CAPTCHA_WIDTH * CAPTCHA_TUNNEL
    NN_OUT_SIZE = CAPTCHA_LEN * CHAR_LEN
    NN_NODE = 1024      # 第一隐藏层的节点大小
    NN_LEARNING_RATE = 0.005
    NN_BATCH_SIZE = 100
    NN_EPOCH = 100      # 训练次数
    NN_SAVE = True


# 验证码装换为 One-hot 向量
def text2vec(text):
    if len(text) != Config.TEXT_LEN:
        return False
    vec = np.zeros(Config.TEXT_LEN * Config.CHAR_LEN)
    for i, c in enumerate(text):
        index = i * Config.CHAR_LEN + Config.CHARS.index(c)
        vec[index] = 1
    return vec


# One-hot 向量转验证码文本
def vec2text(vec):
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)

    vec = np.reshape(vec, [Config.TEXT_LEN, -1])
    text = ''
    for item in vec:
        text += Config.CHARS[np.argmax(item)]
    return text


def read_dir_to_list(p):
    """
    加载目录下的全部文件，经测试在普通机械硬盘下加载 10W 张图片（平均每张1.8kb）耗时 500 s .... =.=
    :param p:
    :return:
    """
    st = clock()
    C.yellow("> start to load image from %s" % p)
    imgs, captchas = [], []
    for _, _, names in os.walk(p):
        C.yellow("> get %d files..." % len(names))
        for n in names:
            captchas.append(n.replace(Config.SUFFIX, ""))

            d = Image.open(os.path.join(p, n))
            imgs.append(np.asarray(d))
    C.yellow("> success load %d images, used %f seconds ..." % (len(imgs), clock()-st))
    return np.asarray(imgs, dtype=np.float32), np.asarray(captchas)


def load_data(p):
    dx, captchas = read_dir_to_list(p)

    def standardize():
        return (dx - dx.mean()) / dx.std()
    dy = []
    for text in captchas:
        dy.append(text2vec(text))
    return standardize(), np.asarray(dy)


X = tf.placeholder(tf.float32, shape=[None, Config.CAPTCHA_HEIGHT, Config.CAPTCHA_WIDTH, Config.CAPTCHA_TUNNEL], name="X")
Y = tf.placeholder(tf.float32, shape=[None, Config.NN_OUT_SIZE], name="Y")

keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(-1, trainable=False)


# 构建模型
def build_model():
    input_x = tf.layers.flatten(X)
    w1 = tf.Variable(tf.random_uniform([Config.NN_IN_SIZE, Config.NN_NODE]))
    b1 = tf.Variable(tf.random_uniform([Config.NN_NODE]))
    layer1 = tf.nn.dropout(tf.add(tf.matmul(input_x, w1), b1), keep_prob=keep_prob)

    w2 = tf.Variable(tf.random_uniform([Config.NN_NODE, Config.NN_OUT_SIZE]))
    b2 = tf.Variable(tf.random_uniform([Config.NN_OUT_SIZE]))
    layer2 = tf.nn.dropout(tf.add(tf.matmul(layer1, w2), b2), keep_prob=keep_prob)

    return layer2


def build_model_dense():
    """
    使用 tf.layers 搭建全连接前向传播模型
    :return:
    """
    input_x = tf.layers.flatten(X)

    y = tf.layers.dense(input_x, Config.NN_NODE, kernel_initializer=tf.initializers.random_uniform)
    y = tf.layers.dropout(y, rate=Config.NN_KEEP_PROB)
    y = tf.layers.dense(y, Config.NN_OUT_SIZE)

    return y


def build_train(y_pred):
    # 损失函数
    loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=y_pred))

    train_op = tf.train.RMSPropOptimizer(Config.NN_LEARNING_RATE).minimize(loss_, global_step)

    # 计算准确率
    # 1. 先进行切分  [60] => [6, 10]
    y_pred_reshape = tf.reshape(y_pred, [-1, Config.CHAR_LEN])
    y_reshape = tf.reshape(Y, [-1, Config.CHAR_LEN])
    # 2. 计算
    max_index_pred = tf.argmax(y_pred_reshape, axis=1)
    max_index = tf.argmax(y_reshape, axis=1)
    max_index_pred = tf.reshape(max_index_pred, [-1, Config.CAPTCHA_LEN])
    max_index = tf.reshape(max_index, [-1, Config.CAPTCHA_LEN])
    fn = lambda x: tf.equal(tf.reduce_mean(tf.cast(x, tf.float32)), 1.)
    is_correct = tf.map_fn(fn, tf.equal(max_index_pred, max_index))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return loss_, train_op, accuracy


prediction = build_model_dense()
loss, train_op, accuracy = build_train(prediction)

saver = tf.train.Saver()


# 执行训练
def train():
    C.bgGreen(">>>> Begin to train ( keep_prob={}, learning_rate={} ) <<<<".format(Config.NN_KEEP_PROB, Config.NN_LEARNING_RATE))
    # 读取数据
    data_x, data_y = load_data(Config.DIR_TRAIN)
    C.green("data_x shape=%s  data_y shape=%s" % (data_x.shape, data_y.shape))

    C.green("split data into Train 80%, Dev 10%, Test 10% ...")
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=40)
    dev_x, test_x, dev_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=40)

    step_train = int(len(train_x) / Config.NN_BATCH_SIZE)
    print("train step = %d" % step_train)

    startOn = clock()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(Config.NN_EPOCH):
            C.yellow("\nbegin the {:<4} epoch training...".format(epoch))
            for batch in range(step_train):
                start = batch * Config.NN_BATCH_SIZE
                end = start + Config.NN_BATCH_SIZE
                dx, dy = train_x[start: end], train_y[start: end]

                _, loss_val, acc_val = session.run([train_op, loss, accuracy], feed_dict={X: dx, Y: dy, keep_prob: Config.NN_KEEP_PROB})

                if batch % 5 == 0:
                    print("\tepoch={:4} batch = {:3} loss= {:<25} accuracy = {:5.2f} %".format(epoch, batch, loss_val, acc_val*100))

            loss_val, acc_val = session.run([loss, accuracy], feed_dict={X: dev_x, Y: dev_y, keep_prob: Config.NN_KEEP_PROB})
            C.yellow("\t[Dev Test] epoch={:5} loss= {:<25} accuracy = {:5.2f} %".format(epoch, loss_val, acc_val*100))

            if Config.NN_SAVE and epoch % 10 == 0:
                saver.save(session, Config.DIR_MODEL+"/captcha")
                C.green("Model saved on %s" % Config.DIR_MODEL)

        C.bgYellow("---- Train Done (used time = %.4f seconds) ----" % (clock() - startOn))
        loss_val, acc_val = session.run([loss, accuracy], feed_dict={X: test_x, Y: test_y, keep_prob: Config.NN_KEEP_PROB})
        C.yellow("\t[Test] loss= {:<25} accuracy = {:5.2f} %".format(loss_val, acc_val * 100))


def predicted(test_dir=Config.DIR_TEST):
    """
    对某个目录下的验证码文件进行预测（文件名即为验证码文本）
    :param test_dir:
    :return:
    """
    C.bgYellow(">>>> Begin to prediction <<<<")

    if not os.path.exists(test_dir) or os.path.isfile(test_dir):
        return C.red("directory not exist: %s" % test_dir)

    # 检查模型
    model = tf.train.get_checkpoint_state(Config.DIR_MODEL)
    if model:
        data_x, data_y = load_data(test_dir)
        print("total %d captcha to be predicted..." % len(data_x))

        with tf.Session() as session:
            saver.restore(session, model.model_checkpoint_path)
            C.green("restore session from %s" % model.model_checkpoint_path)

            y_pred, acc_val = session.run(
                [prediction, accuracy],
                feed_dict={X: data_x, Y: data_y, keep_prob: Config.NN_KEEP_PROB}
            )
            correct_count = 0
            for i in range(len(data_y)):
                origin_code, pred_code = vec2text(data_y[i]), vec2text(y_pred[i])
                correct = origin_code == pred_code
                if correct:
                    correct_count += 1
                    C.green("{} predicted to {} : {}".format(origin_code, pred_code, correct))
                else:
                    C.red("{} predicted to {} : {}".format(origin_code, pred_code, correct))

            C.yellow("\nAccuracy = {:5.2f} % (Correct_count={})".format(acc_val * 100, correct_count))

    else:
        C.red("Model not exist on %s" % Config.DIR_MODEL)


def main():
    C.bgGreen(":::: Welcome to use %s (Version=%s) ::::" % (Config.NAME, Config.VERSION))
    print()
    train()
    # 若需要做预测，请执行 predicted()
    # predicted()


if __name__ == '__main__':
    main()
