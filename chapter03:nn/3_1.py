# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm Community Edition
@project    tensorflow-study
@file       3_1.py
@time       18-5-7 上午10:54
"""
import base
import tensorflow as tf

# 输入为长度为2的一阶张量,数据量为1
x = tf.placeholder(dtype=tf.float32, shape=(None, 2))

w1 = tf.Variable(tf.ones([2, 3]))
w2 = tf.Variable(tf.ones([3, 1]))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(y, feed_dict={x: [[5, 7], [1, 1]]}))
    print(session.run(tf))
