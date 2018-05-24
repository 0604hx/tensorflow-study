# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       captcha_01.py
@time       2018/5/22 9:17
"""
import math
import os
import pickle
import random
import shutil
from os import makedirs
from time import clock

import tensorflow as tf

import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
from sklearn.model_selection import train_test_split

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']    # 验证码中的字符
CHAR_LEN = len(CHARS)
TEXT_LEN = 4

BATCH_TRAIN = 128
BATCH_DEV = 256
BATCH_TEST = 256

PADDING = 'same'
DATA_PATH = "data"
CHECKPOINT_PATH = "models/captcha_01"

LEARN_RATE = 0.001
EPOCH = 1000
KEEP_PROB = 0.5

STEP_PRE_PRINT = 2
STEP_PRE_SAVE = 10
STEP_PRE_DEV = 2        # 每两轮就进行一次检验


def createOnMissing(p):
    if not os.path.exists(p):
        makedirs(p)


# 生成指定文本的验证码，返回 shape=[60, 100, 3] 的数组
def generate(text):
    img = ImageCaptcha(width=len(text) * 25)
    captcha = img.generate(text)
    captcha_img = Image.open(captcha)
    # captcha_img.show()
    return np.array(captcha_img)


# 验证码装换为 One-hot 向量
def text2vec(text):
    if len(text) != TEXT_LEN:
        return False
    vec = np.zeros(TEXT_LEN * CHAR_LEN)
    for i, c in enumerate(text):
        index = i * CHAR_LEN + CHARS.index(c)
        vec[index] = 1
    return vec


# One-hot 向量转验证码文本
def vec2text(vec):
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)

    vec = np.reshape(vec, [TEXT_LEN, -1])
    text = ''
    for item in vec:
        text += CHARS[np.argmax(item)]
    return text


def generate_data(size=10000, filename="captcha_01.pkl"):
    """
    构建数据
    :param filename:
    :param size:
    :return:
    """
    def random_text():
        t = []
        for _ in range(TEXT_LEN):
            t.append(random.choice(CHARS))
        return "".join(t)

    dx, dy = [], []
    for _ in range(size):
        text = random_text()
        dx.append(generate(text))
        dy.append(text2vec(text))

    dx = np.asarray(dx, np.float32)
    dy = np.asarray(dy, np.float32)
    if not filename:
        return dx, dy

    createOnMissing(DATA_PATH)

    # 写入数据
    with open(os.path.join(DATA_PATH, filename), 'wb') as f:
        pickle.dump(dx, f)
        pickle.dump(dy, f)
        print("{} data save on {}".format(size, f))


# 输入数据标准化，若不进行此步操作，训练效果止步不前 =.=
def standardize(x):
    return (x - x.mean()) / x.std()


def load_data(filename="captcha_01.pkl"):
    with open(os.path.join(DATA_PATH, filename), 'rb') as f:
        dx = pickle.load(f)
        dy = pickle.load(f)
    return standardize(dx), dy


data_x, data_y = load_data()
print("data loaded success! x_shape=", data_x.shape, ", y_shape=", data_y.shape)
# print('Data X Length', len(data_x), 'Data Y Length', len(data_y))
# print('Data X Example', data_x[0])
# print('Data Y Example', data_y[0])

# 利用 train_test_split() 方法将数据分为三部分，训练集、开发集、验证集：
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.4, random_state=40)
dev_x, test_x, dev_y, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=40)

train_steps = math.ceil(train_x.shape[0] / BATCH_TRAIN)
dev_steps = math.ceil(dev_x.shape[0] / BATCH_DEV)
test_steps = math.ceil(test_x.shape[0] / BATCH_TEST)

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000).batch(BATCH_TRAIN)
dev_dataset = tf.data.Dataset.from_tensor_slices((dev_x, dev_y)).batch(BATCH_DEV)
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCH_TEST)

# 初始化一个迭代器，并绑定到这个数据集上
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_initializer = iterator.make_initializer(train_dataset)
dev_initializer = iterator.make_initializer(dev_dataset)
test_initializer = iterator.make_initializer(test_dataset)

keep_prob = tf.placeholder(tf.float32, [])
global_step = tf.Variable(-1, trainable=False, name="global_step")

with tf.variable_scope("inputs"):
    # x.shape = [-1, 60, 100, 3]
    x, y_label = iterator.get_next()

y = tf.cast(x, tf.float32)
print("origin=", y.shape)
# 3 层 卷积网络
for _ in range(3):
    y = tf.layers.conv2d(y, filters=32, kernel_size=3, padding=PADDING, activation=tf.nn.relu)
    print(y.shape)
    y = tf.layers.max_pooling2d(y, pool_size=2, strides=2, padding=PADDING)

print(y.shape)

# 2 层全连接网络
y = tf.layers.flatten(y)
print("flatten=", y.shape)
y = tf.layers.dense(y, 1024, activation=tf.nn.relu)
y = tf.layers.dropout(y, rate=keep_prob)
y = tf.layers.dense(y, TEXT_LEN * CHAR_LEN)

y_reshape = tf.reshape(y, [-1, CHAR_LEN])
y_label_reshape = tf.reshape(y_label, [-1, CHAR_LEN])

# 计算损失函数
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_reshape, labels=y_label_reshape))

# 计算准确率
max_index_pred = tf.argmax(y_reshape, axis=-1)
max_index_label = tf.argmax(y_label_reshape, axis=-1)
accuracy = tf.equal(max_index_label, max_index_pred)
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

# 训练
train_op = tf.train.RMSPropOptimizer(LEARN_RATE).minimize(cross_entropy, global_step=global_step)

saver = tf.train.Saver()


def doWithSession(train=True):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        if train:
            print("start to training( old model will be removed )....")
            try:
                shutil.rmtree(CHECKPOINT_PATH)
            except Exception as e:
                print("Could not delete %s : %s" % (CHECKPOINT_PATH, e))

            createOnMissing(CHECKPOINT_PATH)
            for epoch in range(EPOCH):
                tf.train.global_step(session, global_step_tensor=global_step)
                session.run(train_initializer)

                for step in range(int(train_steps)):
                    loss, acc, gstep, _ = session.run([cross_entropy, accuracy, global_step, train_op], feed_dict={keep_prob: KEEP_PROB})

                    if step % STEP_PRE_PRINT == 0:
                        print("[Epoch={:4}] Global Step={:6}  Step={:6} Loss={:<25} Accuracy={:<25}".format(epoch, gstep, step, loss, acc))

                if epoch % STEP_PRE_DEV == 0:
                    print(":::: Call dev test for current model ::::")
                    session.run(dev_initializer)
                    for _ in range(int(dev_steps)):
                        print("Dev Accuracy=", session.run(accuracy, feed_dict={keep_prob: 1.}))

                # 保存模型
                if epoch % STEP_PRE_SAVE == 0:
                    saver.save(session, CHECKPOINT_PATH+"/model", global_step=gstep)
                    print("Saved model : global_step=%d" % gstep)

        else:
            print("testing...")
            feed = {keep_prob: 0.5}

            models = tf.train.get_checkpoint_state(CHECKPOINT_PATH)
            if models:
                saver.restore(session, models.model_checkpoint_path)
                print('Restore model from', models.model_checkpoint_path)

                session.run(test_initializer)
                for step in range(int(test_steps)):
                    print("Test Accuracy=", session.run(accuracy, feed_dict=feed))

                def testWithRandomData(size=20):
                    st = clock()
                    # 开始随机生成验证码并进行验证
                    ax, ay = generate_data(size, None)

                    ax = standardize(ax)
                    dataset = tf.data.Dataset.from_tensor_slices((ax, ay)).batch(size)
                    a_initializer = iterator.make_initializer(dataset)
                    session.run(a_initializer)

                    originTexts = [vec2text(yy) for yy in ay]

                    # 得到预测值
                    y_prediction = session.run(y, feed_dict=feed)
                    predTexts = [vec2text(yy) for yy in y_prediction]

                    print(":::: 本次随机测试 size=%d ::::" % size)
                    print("待识别验证码：", originTexts)
                    print("预测的验证码：", predTexts)

                    correct = [originTexts[i] == predTexts[i] for i in range(size)]
                    print("准确率：", np.mean(correct), " (耗时 %f 秒)" % (clock()-st))
                    print()

                testWithRandomData()
                testWithRandomData(100)
                testWithRandomData(300)
                testWithRandomData(600)
                testWithRandomData(1000)
            else:
                print("No model found on %s" % CHECKPOINT_PATH)


if __name__ == '__main__':
    doWithSession(False)
