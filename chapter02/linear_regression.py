'''
线性回归 推理模型

注：以下内容为个人理解，如有不正确的地方望谅解

线性回归模型所预测的是一个连续值或任意实数
如通过一对数值推算出结果（该数值于传递过来的参数是息息相关的）


我们把从“某些值”预测“另外某个值”的思想称为回归（http://en.wikipedia.org/wiki/Regression_analysis）
回归技术和分类技术关系紧密，通常来说，回归是预测一个数值型数量，比如大小、收入和温度，而分类则预测标号（label）
或类别（category），比如判断邮件是否为“垃圾邮件”，拼图游戏的图案是否为“猫”。

将回归和分类联系在一起是因为两者都可以通过一个（或更多）值预测另一个（或更多）值。为了能够做出预测，两者都需要从
一组输入和输出中学习预测规则。在学习过程中，需要告诉它们问题以及问题的答案。因此，它们都属于有监督学习。


    -- 以上内容摘自《Spark高级数据分析》

示例：
1. 血液脂肪含量与体重、身高有一定的关系
2. 豌豆子代大小与父代大小的关系：父代豌豆大，子代豌豆也大，但要略小于父代豌豆；父代豌豆小，子代豌豆也小，但要略大于父代豌豆。

add on 2017年10月25日20:13:39

代码参考自：https://github.com/backstopmedia/tensorflowbook/blob/master/chapters/04_machine_learning_basics/linear_regression.py
'''

import tensorflow as tf

W = tf.Variable(tf.zeros([2,1]), name="weights")
b = tf.Variable(0., name="bias")

def inference(X):
    '''
    数据推理，根据输入 X 得到输出 Y
    '''
    return tf.matmul(X, W) + b

def loss(X,Y):
    '''
    损失计算
    对于这种简单的模型， 将采用总平方误差， 即模型对每个训练样本的预测值与期望输出之差的平方的总和
    '''
    Y_predicetd = tf.transpose(inference(X))
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicetd))

def inputs():
    '''
    获取训练数据
    '''
    weight_age = [
        [84, 46], [73, 20], [65, 52], [70, 30], [76, 57], 
        [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], 
        [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], 
        [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], 
        [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]
    ]
    blood_fat_content = [
        354, 190, 405, 263, 451, 
        302, 288, 385, 402, 365, 
        209, 290, 346, 254, 395, 
        434, 220, 374, 308, 220, 
        311, 181, 274, 303, 244
    ]

    return tf.to_float(weight_age), tf.to_float(blood_fat_content)


def train(total_loss):
    '''
    训练
    '''
    learning_rate = 0.000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    '''
    模型评估
    '''
    print(sess.run(inference([[50., 20.]]))) # ~ 303
    print(sess.run(inference([[50., 70.]]))) # ~ 256
    print(sess.run(inference([[90., 20.]]))) # ~ 303
    print(sess.run(inference([[90., 70.]]))) # ~ 256

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    X, Y = inputs()
    
    totalLoss = loss(X,Y)
    trainOP = train(totalLoss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    trainingStep = 10000
    for step in range(trainingStep):
        sess.run([trainOP])
        if step % 1000 == 0:
            print("Epoch : {} loss: {}".format(step, sess.run(totalLoss)))

    print("Final model \n\tW= {} \n\tb={}".format(sess.run(W), sess.run(b)))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)

    sess.close()
