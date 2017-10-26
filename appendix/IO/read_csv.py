'''
本例子使用 tensorflow 读取 CSV 格式的文件
官方示例：https://www.tensorflow.org/api_guides/python/reading_data#Reading_from_files

数据文件位于：assets/dataset/train.csv

add on 2017年10月26日14:20:20
'''

import tensorflow as tf
import os

def getCSVFile():
    return os.path.join(os.path.dirname(__file__), "../../assets/dataset/train.csv")

filename = getCSVFile()
fileLen = 891                               # 文件中共有892（其中数据行 891 ）
print("loading ", filename)

# 构建文件名队列，类型为：tensorflow.python.ops.data_flow_ops.FIFOQueue object
fileQueue = tf.train.string_input_producer([filename])

reader = tf.TextLineReader(skip_header_lines=1)
# 每次执行 read 操作都从文件中读取一行数据
key,value = reader.read(fileQueue)

# 我们观察到 train.csv 文件的第二行开始是数据行，格式如下：
# 1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
recordDefaults = [
    [0.],[0.],[0.],[""],
    [""],[0.],[0.],[0.],
    [""],[0.],[""],[""]
]
cols = tf.decode_csv(value, record_defaults=recordDefaults)

with tf.Session() as sess:
    '''
    注意这里要使用 Coordinator 
    TensorFlow提供了两个类来帮助多线程的实现：tf.Coordinator和 tf.QueueRunner。
    从设计上这两个类必须被一起使用。Coordinator类可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常。
    QueueRunner类用来协调多个工作线程同时将多个张量推入同一个队列中。

    详见：http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/threading_and_queues.html
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(fileLen):
        print(sess.run(cols))

    coord.request_stop()
    coord.join(threads)