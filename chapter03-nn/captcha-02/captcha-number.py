# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       captcha-number.py
@time       2018/5/23 16:14

经过 10 轮左右的训练即可得到 98% + 的准确率
实际预测准确率 98% +
"""
import os
from time import clock

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from color import C


class Config(object):
    NAME = "纯数字验证码模型"
    VERSION = "1.0"

    CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # 验证码中的字符
    CHAR_LEN = len(CHARS)
    TEXT_LEN = 6

    DIR_TRAIN = "F:/data/captcha/number-1w"
    DIR_TEST = "F:/data/captcha/number-test"
    DIR_MODEL = "./models"
    SUFFIX = ".jpg"

    CAPTCHA_LEN = 6
    CAPTCHA_WIDTH = CAPTCHA_LEN * 15
    CAPTCHA_HEIGHT = 24
    CAPTCHA_TUNNEL = 3

    NN_KEEP_PROB = 1.
    NN_IN_SIZE = CAPTCHA_HEIGHT * CAPTCHA_WIDTH * CAPTCHA_TUNNEL
    NN_OUT_SIZE = CAPTCHA_LEN * CHAR_LEN
    NN_NODE = 1024      # 第一隐藏层的节点大小
    NN_LEARNING_RATE = 0.001
    NN_BATCH_SIZE = 100
    NN_EPOCH = 100      # 训练次数


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


def build_train(y_pred):
    # 损失函数
    loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=y_pred))

    train_op = tf.train.AdamOptimizer(Config.NN_LEARNING_RATE).minimize(loss_, global_step)

    # 计算准确率
    # 1. 先进行切分  [60] => [6, 10]
    y_pred_reshape = tf.reshape(y_pred, [-1, Config.CHAR_LEN])
    y_reshape = tf.reshape(Y, [-1, Config.CHAR_LEN])
    # 2. 计算
    max_index_pred = tf.argmax(y_pred_reshape, axis=-1)
    max_index = tf.argmax(y_reshape, axis=-1)
    is_correct = tf.equal(max_index_pred, max_index)
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return loss_, train_op, accuracy


prediction = build_model()
loss, train_op, accuracy = build_train(prediction)

saver = tf.train.Saver()


# 执行训练
def train():
    C.bgGreen(">>>> Begin to train <<<<")
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
            C.yellow("\t[Dev Test] epoch={:4} loss= {:<25} accuracy = {:5.2f} %".format(epoch, loss_val, acc_val*100))

            if epoch % 10 == 0:
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

            y_pred, acc_val = session.run([prediction, accuracy], feed_dict={X: data_x, Y:data_y, keep_prob:Config.NN_KEEP_PROB})
            for i in range(len(data_y)):
                origin_code, pred_code = vec2text(data_y[i]), vec2text(y_pred[i])
                correct = origin_code == pred_code
                if correct:
                    C.green("{} predicted to {} : {}".format(origin_code, pred_code, correct))
                else:
                    C.red("{} predicted to {} : {}".format(origin_code, pred_code, correct))

            C.yellow("\nAccuracy = {:5.2f} %".format(acc_val * 100))
    else:
        C.red("Model not exist on %s" % Config.DIR_MODEL)


def main():
    C.bgGreen(":::: Welcome to use %s (Version=%s) ::::" % (Config.NAME, Config.VERSION))
    print()
    # train()
    predicted()


if __name__ == '__main__':
    main()
