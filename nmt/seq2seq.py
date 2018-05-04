# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm
@project    tensorflow-study
@file       seq2seq.py
@time       2018/4/12 10:28

相关教程：https://zhuanlan.zhihu.com/p/27608348
"""

import tensorflow as tf
# 直接用 tf 无法完成 contrib 的代码提示，这里单独引用 contrib 包 =.=
from tensorflow.contrib import layers, rnn

SOURCE = "./data/letters_source.txt"
TARGET = "./data/letters_target.txt"

"""
< PAD>: 补全字符。
< EOS>: 解码器端的句子结束标识符。
< UNK>: 低频词或者一些未遇到过的词等。
< GO>: 解码器端的句子起始标识符。
"""
SPECIAL_WORDS = ['<PAD>', '<UNK>', '<GO>', '<EOS>']


def to_vocab(data):
    """
    构造词汇量
    :param data:
    :return:
    """
    words = list(set([character for line in data for character in line]))
    int_to_words = {idx: word for idx, word in enumerate(SPECIAL_WORDS + words)}
    words_to_int = {word: idx for idx, word in int_to_words.items()}

    return int_to_words, words_to_int


source_data = open(SOURCE, 'r', encoding="utf-8").read().splitlines()
target_data = open(TARGET, 'r', encoding="utf-8").read().splitlines()

source_int_to_word, source_word_to_int = to_vocab(source_data)
target_int_to_word, target_word_to_int = to_vocab(target_data)

# 把原始数据转换成数值向量
source_int = [[source_word_to_int.get(key, source_word_to_int['<UNK>']) for key in line] for line in source_data]
target_int = [[target_word_to_int.get(key, target_word_to_int['<UNK>']) for key in line] + [target_word_to_int['<EOS>']] for line in target_data]


def get_encoder_layer(input_data, rnn_size, num_layers, source_seq_len, source_vocab_size, embedding_size):
    """
    构造 encoder 层
    :param input_data:
    :param rnn_size:
    :param num_layers:
    :param source_seq_len:
    :param source_vocab_size:
    :param embedding_size:
    :return:
    """
    encoder_embed_input = layers.embed_sequence(input_data, source_vocab_size, embedding_size)

    # build RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    encoder_out, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, source_seq_len, dtype=tf.float32)
    return encoder_out, encoder_state


def process_decoder_input(data, vocab_to_int, batch_size):
    """
    处理 decoder 的输入： 补充 <GO> 并删除最后一个字符
    tf.strided_slice 函数的说明： https://blog.csdn.net/banana1006034246/article/details/75092388

    :param data:
    :param vocab_to_int:
    :param batch_size:
    :return:
    """
    # cut掉最后一个字符
    encoding = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    # 其中tf.fill(dims, value)参数会生成一个dims形状并用value填充的tensor
    # 举个栗子：tf.fill([2,2], 7) => [[7,7], [7,7]]。tf.concat()会按照某个维度将两个tensor拼接起来
    go_fill = tf.fill([batch_size, 1], vocab_to_int['<GO>'])
    return tf.concat([go_fill, encoding], 1)


with tf.Session() as session:
    x = tf.fill([2, 1], 1)
    print(session.run(x))

    y = tf.concat([x, [[2], [3]]], 1)
    print(session.run(y))
