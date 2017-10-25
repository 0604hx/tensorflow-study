'''
变量练习

add on 2017年10月23日21:58:21
'''

import tensorflow as tf

# 创建一个初始值为 1 的 Variable
a = tf.Variable(1)

with tf.Session() as sess:
    # 先执行 initialize_all_variables() （一个 tensorflow OP），对全部的 Variable 进行初始化
    # 新版本的 tf API 推荐使用： global_variables_initializer() 进行 Variable 的初始化
    sess.run(tf.global_variables_initializer()) 
    print(sess.run(a.assign_add(5)))    # 变量自增 5 ，result = 6
    print(sess.run(a.assign(a * 5)))    # 变量乘 5， result = 30
    print(sess.run(a.assign_sub(10)))   # 变量自减 10，result = 20