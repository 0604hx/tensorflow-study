# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm Community Edition
@project    tensorflow-study
@file       3_2.py
@time       18-5-7 下午3:12

构造神经网络,求解 W:
y = x * w
其中 y = shape(None, 1)
    x = shape(Node, 3), 三项指标分别为 身高/体重/性别
    w = shape(3, 1)

预设 w = [ [0.6], [0.3], [0.1] ]
"""
import base
import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
SEED = 123456


def initData(size=100):
    """
    准备数据
    :return:
    """
    rdn = np.random.RandomState()
    # 构建身高数据,从 130.0 到 200.0
    heights = [[130.0 + h * 70] for h in rdn.rand(size)]
    # 体重,从 70.0 到 180.0
    weights = [[70.0 + h * 110] for h in rdn.rand(size)]
    # 性别,0=男性, 1=女性
    sexs = [[0. if h <= 0.57 else 1.] for h in rdn.rand(size)]
    dx = np.column_stack((heights, weights, sexs))
    # 计算结果
    dy = [[h * 0.6 + w * 0.3 + s * 0.1] for (h, w, s) in dx]
    return dx, dy


def run():
    # 1. 准备数据
    data_size = 10000
    X, Y = initData(data_size)

    # 设计前向传播
    x = tf.placeholder(dtype=tf.float32, shape=(None, 3), name="x")
    y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")

    # 此处定义了一个2层(4节点的隐藏层 + 1输出层)的神经网络
    w1 = tf.Variable(tf.ones((3, 4)))
    w2 = tf.Variable(tf.ones((4, 1)))

    y_ = tf.matmul(tf.matmul(x, w1), w2)

    # 设计反向传播
    loss = tf.reduce_mean(tf.square(y-y_))
    global_step = tf.Variable(0, trainable=False)
    # 使用指数下降学习率
    learning_rate = tf.train.exponential_decay(0.05, global_step, BATCH_SIZE, 1.1, staircase=False) #tf.constant(0.1, dtype=tf.float32)
    #使用梯度下滑优化器,训练次数跟 learning_rate 有关, 0.0000024
    # train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 用 adam 优化器达到更好的效果, 0.1
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            start = i * BATCH_SIZE % data_size
            end = start + BATCH_SIZE
            # print("range= %d, %d" % (start, end))
            dx, dy = X[start: end], Y[start: end]
            sess.run(train, feed_dict={x: dx, y: dy})
            if i % 50 == 0:
                print("[{:5} training] loss = {:<20} learning_rate={:<20} global_step={:4}".format(
                    i,
                    sess.run(loss, feed_dict={x: dx, y: dy}),
                    sess.run(learning_rate),
                    sess.run(global_step)
                ))

        print("w1 = ", sess.run(w1))
        print("w2 = ", sess.run(w2))

        print("======== 开始使用模型 ========")
        tx, ty = initData(5)
        print("预测值=", sess.run(y_, feed_dict={x: tx}))
        print("实际值=", sess.run(tf.constant(ty)))


if __name__ == '__main__':
    run()
