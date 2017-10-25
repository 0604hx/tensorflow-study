'''
使用不同的名称作用域分割数据流图，并保存到 TensorBoard 中

add on 2017年10月24日10:04:49
'''

import tensorflow as tf

# 创建数据图
graph = tf.Graph()

with graph.as_default():
    input01 = tf.placeholder(tf.int32, name="input01")
    input02 = tf.placeholder(tf.int32, name="input02")
    const = tf.constant(10)

    with tf.name_scope("Transformation"):

        with tf.name_scope("A"):
            a_mul = tf.multiply(input01, const)
            a_out = tf.subtract(a_mul, input01)

        with tf.name_scope("B"):
            b_mul = tf.multiply(input02, const)
            b_out = tf.subtract(b_mul, input02)

        with tf.name_scope("C"):
            c_div = tf.div(a_out, b_out)
            c_out = tf.add(c_div, const)

        with tf.name_scope("D"):
            d_div = tf.div(b_out, a_out)
            d_out = tf.add(d_div, const)

    out = tf.maximum(c_out,d_out)
    
    writer = tf.summary.FileWriter('./_board', graph=graph)
    writer.close()
    print("graph write successed! You can run 'tensorboard --logdir=\"./_board\"' to watch graph.")