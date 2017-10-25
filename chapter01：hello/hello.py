"""
第一章：tensorflow入门程序

绘制以下内容：
    a   input 节点，5
    b   input 节点，3
    c   add 操作
    最后保存 graph 到 ../graph/01=hello
"""

import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.add(a,b,"add_c")

graph = tf.Graph()
session = tf.Session()

result = session.run(c)
print(result)

session.close()

