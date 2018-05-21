# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       nn_2L.py
@time       2018/5/17 11:18

使用 2 层全连接神经网络进行训练
"""
import base
import tensorflow as tf

from helper import IMAGE_WIDTH, IMAGE_HEIGHT, CHAR_LEN, TEXT_LEN, next_batch, vec_to_text

IN_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH
OUT_SIZE = TEXT_LEN * CHAR_LEN
BATCH_SIZE = 128
LAYER_SIZE_ONE = 256

X = tf.placeholder(dtype=tf.float32, shape=[None, IN_SIZE])
Y = tf.placeholder(dtype=tf.float32, shape=[None, OUT_SIZE])
keep_prob = tf.placeholder(tf.float32)  # dropout


def build_weight():
    # 第一层神经有 120 个节点
    w1 = tf.Variable(tf.random_uniform([IN_SIZE, LAYER_SIZE_ONE]))
    b1 = tf.Variable(tf.random_uniform([LAYER_SIZE_ONE]))
    layer1 = tf.nn.leaky_relu(tf.matmul(X, w1) + b1)

    w2 = tf.Variable(tf.random_uniform([LAYER_SIZE_ONE, OUT_SIZE]))
    b2 = tf.Variable(tf.random_uniform([OUT_SIZE]))
    layer2 = tf.nn.leaky_relu(tf.matmul(layer1, w2) + b2)
    return layer2

    # w3 = tf.Variable(tf.random_uniform([OUT_SIZE, OUT_SIZE]))
    # b3 = tf.Variable(tf.random_uniform([OUT_SIZE]))
    # layer3 = tf.nn.dropout(tf.matmul(layer2, w3) + b3, keep_prob)
    #
    # return layer3


def build_train(prediction):
    global_step = tf.Variable(0, trainable=False)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction))
    loss = tf.reduce_mean(tf.squared_difference(Y, prediction))

    train_step = tf.train.AdamOptimizer(learning_rate=0.25).minimize(loss, global_step)
    return loss, train_step


prediction = build_weight()
loss, train_step = build_train(prediction)

# predictionis_correct = tf.equal(vec_, tf.argmax(prediction, 1))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# 开始训练
with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for i in range(1000):
        dx, dy = next_batch()

        feed = {X: dx, Y: dy}
        _, loss_val = session.run([train_step, loss], feed_dict=feed)
        print("[epoch {:5}] loss={}".format(i+1, loss_val))

    print("test-----")
    tx, ty = next_batch(1)
    tp = session.run(prediction, feed_dict={X: tx, Y: ty})
    print(tx, ty)
    print(tp)
    print(vec_to_text(tp[0]))
