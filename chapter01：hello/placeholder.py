'''
placeholder 练习程序

利用 tf 的 placeholder OP 创建占位符，然后在 session.run 时传递 feed_dict 参数赋值

add on 2017年10月23日21:47:16
'''

import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.int32, name="placeholder")

b = tf.reduce_prod(a, name="prod")
c = tf.reduce_sum(a, name="sum")

d = tf.add(b,c , name="add")

with tf.Session() as sess:
    result = sess.run(d, feed_dict={a: np.array([1,2,3,4,5,6],dtype=np.int32)})
    print(result)