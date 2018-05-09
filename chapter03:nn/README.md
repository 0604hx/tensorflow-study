# Neural Network(NN)
> add on `2018年05月07日10:21:29`

启蒙教程:[人工智能实践：Tensorflow笔记](https://www.icourse163.org/course/PKU-1002536002)


## 前向传播

简单来说,`前向传播`是指由原始数据(`输入`)经过既定的计算过程得到`输出`的过程.

![](imgs/nn_01.png)

如:
```python
import tensorflow as tf

# 输入为长度为2的一阶张量,数据量为1
x = tf.constant([[5.0, 7.0]])

w1 = tf.Variable(tf.ones([2, 3]))
w2 = tf.Variable(tf.ones([3, 1]))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print(session.run(y))d
```

## 反向传播