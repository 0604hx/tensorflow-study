import tensorflow as tf

"""
TensorFlow 测试代码
"""

a = tf.constant(5,name="input_a")
b = tf.constant(3,name="input_b")

c = tf.add(a,b,"add_c")

session = tf.Session()

print(session.run(c))


v = tf.Variable(0,"counter")
one = tf.constant(1)
add_v = tf.add(v,one)

update_counter = tf.assign(v, add_v)

session.run(tf.global_variables_initializer())
for _ in range(10):
    print(session.run(update_counter))