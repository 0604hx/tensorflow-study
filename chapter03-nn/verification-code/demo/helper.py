# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       helper.py
@time       2018/5/17 10:46

负责读取图片数据及预处理
"""
import os
from random import choice

import numpy as np
from PIL import Image

NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']    # 验证码中的字符
CHAR_LEN = len(NUMBERS)
DATA_DIR = "./train"                                            # 数据目录
SUFFIX = ".gif"
IMAGE_HEIGHT = 24                                               # 验证码图片的高度
IMAGE_WIDTH = 60                                                # 验证码图片的宽度
TEXT_LEN = 4                                                    # 验证码长度


def read_dir_to_list(parent=DATA_DIR):
    imgs = []
    for _, _, fnames in os.walk(parent):
        for f in fnames:
            imgs.append(f.replace(SUFFIX, ""))
    return imgs


images = read_dir_to_list()


def random_image():
    """
    获取一个随机图片
    :return: 验证码、图片像素数组
    """
    f = choice(images)
    captcha_img = Image.open(DATA_DIR+"/"+f+SUFFIX)
    return f, np.array(captcha_img)


def text_to_vec(text):
    """
    文字转向量
    :param text:
    :return:    一维数组，长度为 4 * 10
    """
    if len(text) != TEXT_LEN:
        raise Exception("captcha length must be %d" % TEXT_LEN)
    vec = np.zeros(TEXT_LEN * CHAR_LEN)
    for i, v in enumerate(text):
        # 计算字符的位置
        loc = ord(v) - ord('0')
        vec[i*CHAR_LEN + loc] = 1
    return vec


def vec_to_text(vec):
    """
    向量转文字
    :param vec:
    :return:
    """
    idx = np.nonzero(vec)[0]
    text = []
    for i, c in enumerate(idx):
        text.append(NUMBERS[c % CHAR_LEN])
    return "".join(text)


def next_batch(batch_size=128):
    """
    获取批次数据
    :param batch_size:
    :return:
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, TEXT_LEN * CHAR_LEN])

    for i in range(batch_size):
        # 获取随机验证码
        text, image = random_image()
        batch_y[i, :] = text_to_vec(text)
        batch_x[i, :] = image.flatten()     # 合并为一维数组

    return batch_x.astype(np.float32), batch_y.astype(np.float32)


if __name__ == '__main__':
    dx, dy = next_batch(1)
    print(vec_to_text(dy[0]))
    vec = text_to_vec("6513")
    print(vec_to_text(vec))
