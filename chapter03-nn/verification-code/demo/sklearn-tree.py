# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       train002.py
@time       2018/5/14 18:25

sklearn 实现决策树：https://www.cnblogs.com/pinard/p/6056319.html

再探决策树算法之利用sklearn进行决策树实战: https://www.cnblogs.com/AlwaysT-Mac/p/6647192.html
"""
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

from sklearn import tree


def buildMatrix(row=20, col=5, min_value=20, max_value=300):
    return [[random.randint(min_value, max_value) for _ in range(col)] for _ in range(row)]


def buildTarget(row=20):
    """
    构建一维数组，值为 0 或 1
    :param row:
    :return:
    """
    return [random.randint(0, 1) for _ in range(row)]


# 生成 40 行，5 列的随机数，取值范围 20 到 300
X = buildMatrix(40, 5)
Y = buildTarget(40)

scale = StandardScaler()
# 对数据进行标准化处理
X = scale.fit_transform(X)

# 把原始数据按照 7:3 进行分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# 训练决策树（使用 基尼系数 构造）
dt = tree.DecisionTreeClassifier(max_depth=6)
dt.fit(x_train, y_train)

# 使用 classification_report 进行真实值与预测值的对比
print("训练集准确率：\n%s " % classification_report(y_train, dt.predict(x_train)))
print("测试集准确率：\n%s " % classification_report(y_test, dt.predict(x_test)))

# 保存到文件
with open("tree.dot", 'w') as f:
    tree.export_graphviz(dt, f)
    print("决策树保存到 %s" % f.name)
