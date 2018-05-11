# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       mnist.py
@time       2018/5/10 9:40

使用全连接神经网络进行手写数值的识别
1. 数据加载
2. 设计前向传播神经网络
3. 设计反向传播
4. 训练（20轮）

LOGS:
* 使用全连接一层（没有隐藏层），Adam 优化器，非滑动平均
    学习率 0.01，准确率： 93%

* 使用全连接一层（没有隐藏层），Adam 优化器，滑动平均
    学习率 0.01，准确率： 93.1%

"""
import base
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

VALID_SIZE = 20         # 评估数据大小
INPUT_SIZE = 784        # 28 * 28 像素
OUTPUT_SIZE = 10        # 结果为 0 - 9
BATCH_SIZE = 1000       # 一次喂入神经网络的数据
EPOCH = 20              # 训练总次数

# 滑动平均
MOVING_ABLE = False
MOVING_DECAY = 0.99     # 滑动平均率

LEARNING_RATE = 0.01

LOSS_WITH_CROSS = True  # 使用交叉熵计算损失值

data = input_data.read_data_sets("../assets/MNIST", one_hot=True)

print(
    """
    Size of:
        Training    : {}
        Test        : {}
        Validation  : {}
    
    First 3 labels in Test:
        {}
    """.format(
        len(data.train.labels),
        len(data.test.labels),
        len(data.validation.labels),
        data.test.labels[0:3]
    ))


def weights(x):
    """

    :param x: 输入
    :return:
    """
    w1 = tf.Variable(tf.random_uniform([INPUT_SIZE, OUTPUT_SIZE]))
    b1 = tf.Variable(tf.zeros(shape=[OUTPUT_SIZE]))
    return tf.nn.softmax(tf.matmul(x, w1) + b1)


# 定义输入及输出
x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SIZE])
y = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SIZE])

# 预测值
prediction = weights(x)

# 优化器
global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction) if LOSS_WITH_CROSS
    else tf.squared_difference(y, prediction)
)

if MOVING_ABLE:
    print("using ExponentialMovingAverage (decay=%f)..." % MOVING_DECAY)
    variable_avr = tf.train.ExponentialMovingAverage(MOVING_DECAY, global_step)
    variable_avr_op = variable_avr.apply(tf.trainable_variables())

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_op, variable_avr_op]):
        train_step = tf.no_op(name="train")
else:
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for ei in range(EPOCH):
        for i in range(data.train.num_examples // BATCH_SIZE):
            dx, dy = data.train.next_batch(BATCH_SIZE)
            _, loss_val = sess.run([train_step, loss], feed_dict={x: dx, y: dy})
            # print("loss= {}", loss_val)

        # 进行正确率测试
        print("[EPOCH={:4}] accuracy= {}".format(ei+1, sess.run(accuracy, feed_dict={x: data.test.images, y: data.test.labels})))

    print("\n:::::: try to validation( size = %d )" % VALID_SIZE)
    tx, ty = data.validation.next_batch(VALID_SIZE)
    predictions, acc_val = sess.run([prediction, accuracy], feed_dict={x: tx, y: ty})
    print("prediction=", sess.run(tf.argmax(predictions, 1)))
    print("label=", ty)
    print("\n:::::: validation result: accuracy = %f" % acc_val)