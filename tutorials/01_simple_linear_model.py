# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       01_simple_linear_model.py
@time       2018/1/19 11:51
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data

# plot the results
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 载入数据：MNIST数据集大约有12MB，如果给定的地址里没有文件，它将自动下载。
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

print(
    """
Size of:
    Training    : {}
    Test        : {}
    Validation  : {}
    
First 5 labels in Test:
{}
""".format(
        len(data.train.labels),
        len(data.test.labels),
        len(data.validation.labels),
        data.test.labels[0:5]
    ))

# 现在我们可以看到测试集中前面五张图像的类别。将这些与上面的One-Hot编码的向量进行比较。
# 例如，第一张图像的类别是7，对应的在One-Hot编码向量中，除了第7个元素其他都为零。

data.test.cls = np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[0:5])

# 下面开始定义常量
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of classes, one class for each of 10 digits.
num_classes = 10


def draw_images(images, cls, cls_pred=None, title=None):
    """
    用来绘制图像的帮助函数：这个函数用来在3x3的栅格中画9张图像，然后在每张图像下面写出真实的和预测的类别。
    :param images:      待绘制的图像
    :param cls:         分类
    :param cls_pred:    预测分类
    :return:
    """
    assert len(images) == len(cls) == 9

    fig, axes = plt.subplots(3, 3)
    if title is not None:
        fig.text(0, 0, title)

    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="binary")

        # 显示真实、预测的分类
        x_label = "True: {}".format(cls[i]) if cls_pred is None else "True: {}, Pred: {}".format(cls[i], cls_pred[i])

        ax.set_xlabel(x_label)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# 绘制几张图片看看效果
draw_images(data.test.images[0:9], data.test.cls[0:9], title="打印测试数据的 9 张图片")

'''开始定义tensorflow'''

# 首先我们为输入图像定义placeholder变量。这让我们可以改变输入到TensorFlow图中的图像。
# 这也是一个张量（tensor），代表一个多维向量或矩阵。
# 数据类型设置为`float32`，形状设为`[None, img_size_flat]`，`None`代表tensor可能保存着任意数量的图像，
# 每张图象是一个长度为`img_size_flat`的向量。
x = tf.placeholder(tf.float32, [None, img_size_flat])

# 接下来我们为输入变量`x`中的图像所对应的真实标签定义placeholder变量。
# 变量的形状是`[None, num_classes]`，这代表着它保存了任意数量的标签，每个标签是长度为`num_classes`的向量，本例中长度为10。
y_true = tf.placeholder(tf.float32, [None, num_classes])

# 最后我们为变量`x`中图像的真实类别定义placeholder变量。
# 它们是整形，并且这个变量的维度设为`[None]`，代表placeholder变量是任意长的一维向量。
y_true_cls = tf.placeholder(tf.int64, [None])

# 除了上面定义的那些给模型输入数据的变量之外，TensorFlow还需要改变一些模型变量，使得训练数据的表现更好。
# 第一个需要优化的变量称为权重`weight`，TensorFlow变量需要被初始化为零，它的形状是`[img_size_flat, num_classes]`，
# 因此它是一个`img_size_flat`行、`num_classes`列的二维张量（或矩阵）。
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))

# 第二个需要优化的是偏差变量`biases`，它被定义成一个长度为`num_classes`的1维张量（或向量）。
biases = tf.Variable(tf.zeros([num_classes]))

# 这个最基本的数学模型将placeholder变量`x`中的图像与权重`weight`相乘，然后加上偏差`biases`。
# 结果是大小为`[num_images, num_classes]`的一个矩阵，由于`x`的形状是`[num_images, img_size_flat]`
# 并且 `weights`的形状是`[img_size_flat, num_classes]`，因此两个矩阵乘积的形状是`[num_images, num_classes]`，
# 然后将`biases`向量添加到矩阵每一行中。
logits = tf.matmul(x, weights) + biases

# 现在`logits`是一个 `num_images` 行`num_classes`列的矩阵，第$i$行第$j$列的那个元素代表着第$i$张输入图像有多大可能性是第$j$个类别。
# 然而，这是很粗略的估计并且很难解释，因为数值可能很小或很大，因此我们想要对它们做归一化，
# 使得`logits`矩阵的每一行相加为1，每个元素限制在0到1之间。这是用一个称为softmax的函数来计算的，结果保存在`y_pred`中。
y_pred = tf.nn.softmax(logits)

# 可以从`y_pred`矩阵中取每行最大元素的索引值，来得到预测的类别。
y_pred_cls = tf.argmax(y_pred, dimension=1)

'''损失优化函数'''

# 为了使模型更好地对输入图像进行分类，我们必须改变`weights`和`biases`变量。首先我们需要比较模型的预测输出`y_pred`和期望输出`y_true`，来了解目前模型的性能如何。
# 交叉熵（cross-entropy）是一个在分类中使用的性能度量。交叉熵是一个常为正值的连续函数，如果模型的预测值精准地符合期望的输出，它就等于零。
# 因此，优化的目的就是最小化交叉熵，通过改变模型中`weights`和`biases`的值，使交叉熵越接近零越好。
# TensorFlow有一个内置的计算交叉熵的函数。需要注意的是它使用`logits`的值，因为在它内部也计算了softmax。
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

# 现在，我们已经为每个图像分类计算了交叉熵，所以有一个当前模型在每张图上的性能度量。
# 但是为了用交叉熵来指导模型变量的优化，我们需要一个额外的标量值，因此我们简单地利用所有图像分类交叉熵的均值。
cost = tf.reduce_mean(cross_entropy)

'''优化方法'''

# 现在，我们有一个需要被最小化的损失度量，接着我们可以创建优化器。在这种情况中，用的是梯度下降的基本形式，步长设为0.5。
# 优化过程并不是在这里执行。实际上，还没计算任何东西，我们只是往TensorFlow图中添加了优化器，以便之后的操作。
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

'''性能度量'''

# 我们需要另外一些性能度量，来向用户展示这个过程。
# 这是一个布尔值向量，代表预测类型是否等于每张图片的真实类型。
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# 上面先将布尔值向量类型转换成浮点型向量，这样子False就变成0，True变成1，然后计算这些值的平均数，以此来计算分类的准确度。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

'''运行 TensorFlow'''

session = tf.Session()
session.run(tf.global_variables_initializer())

# 在训练集中有50,000张图。用这些图像计算模型的梯度会花很多时间。因此我们利用随机梯度下降的方法，它在优化器的每次迭代里只用到了一小部分的图像。
batch_size = 100


# 函数执行了多次的优化迭代来逐步地提升模型的`weights`和`biases`。在每次迭代中，从训练集中选择一批新的数据，然后TensorFlow用这些训练样本来执行优化器。
def optimize(num_itera):
    for i in range(num_itera):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_batch = data.train.next_batch(batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        # Note that the placeholder for y_true_cls is not set
        # because it is not used during training.
        feed_dict = {x: x_batch, y_true: y_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict)


feed_dict_test = {x: data.test.images,
                  y_true: data.test.labels,
                  y_true_cls: data.test.cls}


def print_accuracy():
    # Use TensorFlow to compute the accuracy.
    acc, corr_pred = session.run([accuracy, correct_prediction], feed_dict=feed_dict_test)
    print(corr_pred)
    # Print the accuracy.
    print("Accuracy on test-set: {0:.1%}".format(acc))


def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')


def plot_example_errors(op_num):
    # Use TensorFlow to get a list of boolean values
    # whether each test-image has been correctly classified,
    # and a list for the predicted class of each image.
    correct, cls_pred = session.run([correct_prediction, y_pred_cls],
                                    feed_dict=feed_dict_test)

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    acc = session.run(accuracy, feed_dict=feed_dict_test)

    # Plot the first 9 images.
    draw_images(images=images[0:9],
                cls=cls_true[0:9],
                cls_pred=cls_pred[0:9], title="{} 次训练后 准确率={:.1%} ".format(op_num, acc))


def draw_weights():
    # Get the values for the weights from the TensorFlow variable.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Create figure with 3x4 sub-plots,
    # where the last 2 sub-plots are unused.
    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Only use the weights for the first 10 sub-plots.
        if i < 10:
            # Get the weights for the i'th digit and reshape it.
            # Note that w.shape == (img_size_flat, 10)
            image = w[:, i].reshape(img_shape)

            # Set the label for the sub-plot.
            ax.set_xlabel("Weights: {0}".format(i))

            # Plot the image.
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


for i in [0, 10, 100, 1000]:
    optimize(i)
    print_accuracy()
    plot_example_errors(i)

session.close()
