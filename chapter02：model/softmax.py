'''
softmax 分类

add on 2017年10月27日15:49:21
'''
import sys
sys.path.append(".")

import os
import tensorflow as tf
# from common.IO import readCSV

# 定义模型
W = tf.Variable(tf.zeros([4,3]), name="weights")
b = tf.Variable(tf.zeros([3]), name="bias")

def readCSV(batch_size, file_name, record_defaults):
    '''
    读取指定CSV文件（目前仅支持单个文件）
    '''
    filename_queue = tf.train.string_input_producer([file_name])

    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # batch actually reads the file and loads "batch_size" rows in a single tensor
    return tf.train.shuffle_batch(decoded,
                                  batch_size=batch_size,
                                  capacity=batch_size * 50,
                                  min_after_dequeue=batch_size)

def inputs():
    '''
    获取 assets/dataset/iris.data 的输入

    iris 数据集中的列描述：
        Attribute Information:
            1. sepal length in cm 
            2. sepal width in cm 
            3. petal length in cm 
            4. petal width in cm 
            5. class: 
            -- Iris Setosa 
            -- Iris Versicolour 
            -- Iris Virginica
    '''
    filename = os.path.join(os.path.dirname(__file__), "../assets/dataset/iris.data")
    sepalLen, sepalWidth, petalLen, petalWidth, label = readCSV(100, filename, [[0.],[0.],[0.],[0.],[""]])

    # 把类别名称转换为从 0 开始计数的类别索引，这里只有三个分类，则为 0 到 2 之间的整数
    labelNumber = tf.to_int32(
        tf.argmax(
            tf.to_int32(
                tf.stack(
                    [
                        tf.equal(label, ['Iris-setosa']),
                        tf.equal(label, ['Iris-versicolor']),
                        tf.equal(label, ['Iris-virginica'])
                    ]
                )
            ),
            0
        )
    )

    # 将所关心的所有特征装入单个矩阵中，然后对矩阵转置，使其每行对应一个样本，每列对应一个特征
    # 通过 stack 函数得到的格式为：
    # shape(4,100)
    #  [
    #   [1.9, .........],
    #   [3.4, .........],
    #   [5.9, .........],
    #   [0.2, .........],
    #  ]
    # 然后转置为：
    # shape(100,4): [ [ 5.0999999 ,  3.79999995,  1.60000002,  0.2       ]]
    features = tf.transpose(tf.stack([sepalLen, sepalWidth, petalLen, petalWidth]))
    
    return features, labelNumber

def combine_inputs(X):
    return tf.matmul(X, W) + b

def inference(X):
    '''
    推理函数
    '''
    return tf.nn.softmax(combine_inputs(X))

def loss(X,Y):
    '''
    损失函数
    在代码层面， TensorFlow为softmax交叉熵函数提供了两个实现版本： 一个版本针对训练集中每个样本只对应单个类别专门做了优化。 
    例如， 训练数据可能有一个类别值或者是“dog”， 或者是“person”或“tree”。
     这个函数是tf.nn.sparse_softmax_cross_entropy_with_logits。
    '''
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))

def train(total_loss):
    '''
    继续使用梯度下降优化器来训练
    '''
    return tf.train.GradientDescentOptimizer(0.0001).minimize(total_loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 获取输入数据，shape(100,4) 的特征数据
    # Y 是分类结果，shape(100, )，取值范围为 0-2
    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(1000):
        sess.run([train_op])
        if i % 10 == 0:
            print("loss=", sess.run([W]))

    coord.request_stop()
    coord.join(threads)
    sess.close()
