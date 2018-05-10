'''
线性回归简单练习

add on 2017年10月26日09:52:22
'''
import tensorflow as tf
import math

X = tf.placeholder(tf.float32, name="X")
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
Y = tf.placeholder(tf.float32, name="Y")

y = W * X + b

# 设置优化器
optimizer = tf.train.GradientDescentOptimizer(0.00000001)
loss = tf.square(Y-y)
train = optimizer.minimize(loss)

# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())

steps = 1000
for i in range(steps):
    feed = {X: [i], Y : [3. * i]}
    sess.run(train, feed_dict=feed)

    w_ = sess.run(W)
    # 如果 W 得到的值是 nan 可以结束训练了
    if math.isnan(w_):
        print("(step={:4}) Now W is NAN..".format(i))
        break

    if(i % 100 == 0):
        print("{:4}  W={}\t b={}\t".format(i,  w_, sess.run(b))," loss=",sess.run(loss,feed_dict=feed))

sess.close()