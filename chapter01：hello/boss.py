'''
tensorflow 组件的综合使用

add on 2017年10月24日17:40:19
'''

import tensorflow as tf
from numpy import random

graph = tf.Graph()

with graph.as_default():

    with tf.name_scope("Variables"):
        '''
        变量作用域，分别存储了数据流图运行的次数、总和
        '''
        globalStep = tf.Variable(0,dtype=tf.int32,trainable=False, name="globalStep")
        globalSum = tf.Variable(0.0,dtype=tf.float32, trainable=False, name="globalSum")

    with tf.name_scope("Transformations"):
        '''
        变换作用域
        输入为两个任意维度的张量，分别进行乘积、求和操作后，再相加，最后得到结果
        '''
        with tf.name_scope("input"):
            input01 = tf.placeholder(tf.float32, shape=[None], name="input01")

        with tf.name_scope("layer01"):
            tmpA = tf.reduce_prod(input01, name="prod")
            tmpB = tf.reduce_sum(input01, name="sum")

        with tf.name_scope("output"):
            output = tf.add(tmpA, tmpB, name="output")

    with tf.name_scope("Update"):
        '''
        更新 Variable
        对于 GlobalStep 进行自增1，globalSum += output
        '''
        print("run variables update ...")

        steps = globalStep.assign_add(1)
        total = globalSum.assign_add(output)

    with tf.name_scope("Summary"):
        avg = tf.div(total, tf.cast(steps, tf.float32), name="avg")

        tf.summary.scalar('Ouput', output)
        tf.summary.scalar('Sum of outputs:', total)
        tf.summary.scalar('Avg of output:', avg)

    with tf.name_scope("Global"):
        init  = tf.global_variables_initializer()
        merged_summary = tf.summary.merge_all()

sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./_board', graph=graph)
sess.run(init)

def run(input_tensor):
    '''
    执行数据流图
    '''
    feed_dict = {input01:input_tensor}
    _,step,summary = sess.run([output, steps, merged_summary], feed_dict=feed_dict)
    writer.add_summary(summary, global_step=step)


for index in range(1000):
    size = random.randint(10)+1
    datas = random.randint(50, size=size)
    print("create random data index={:0>4} tensor len={:^3} data={}".format(index, size,datas))
    run(datas)

writer.flush()
writer.close()