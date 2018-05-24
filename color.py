# encoding: utf-8


"""
@author     0604hx
@license    MIT 
@contact    zxingming@foxmail.com
@site       https://github.com/0604hx
@software   PyCharm Community Edition
@project    terminal-dns
@file       color.py
@time       18-1-11 下午9:28

颜色显示

实现过程：
      终端的字符颜色是用转义序列控制的，是文本模式下的系统显示功能，和具体的语言无关。
      转义序列是以ESC开头,即用\033来完成（ESC的ASCII码用十进制表示是27，用八进制表示就是033）。

书写格式：
     开头部分：\033[显示方式;前景色;背景色m + 结尾部分：\033[0m
     注意：开头部分的三个参数：显示方式，前景色，背景色是可选参数，可以只写其中的某一个；
     另外由于表示三个参数不同含义的数值都是唯一的没有重复的，所以三个参数的书写先后顺序没有固定要求，系统都能识别；
     但是，建议按照默认的格式规范书写。
     对于结尾部分，其实也可以省略，但是为了书写规范，建议\033[***开头，\033[0m结尾。

数值表示的参数含义：
显示方式: 0（默认）、1（高亮）、22（非粗体）、4（下划线）、24（非下划线）、 5（闪烁）、25（非闪烁）、7（反显）、27（非反显）
前景色:   30（黑色）、31（红色）、32（绿色）、 33（黄色）、34（蓝色）、35（洋 红）、36（青色）、37（白色）
背景色:   40（黑色）、41（红色）、42（绿色）、 43（黄色）、44（蓝色）、45（洋 红）、46（青色）、47（白色）


这里只提供绿色、红色两种颜色的封装
"""

ENABLE = True


class Color(object):

    # def green(self,str):
    #     return '\033[0;32m{}\033[0m'.format(str) if ENABLE else str
    #
    # def bgGreen(self,str):
    #     return '\033[0;30;42m{}\033[0m'.format(str) if ENABLE else str
    #
    # def red(self, str):
    #     return '\033[0;31m{}\033[0m'.format(str) if ENABLE else str
    #
    # def bgRed(self, str):
    #     return '\033[0;30;41m{}\033[0m'.format(str) if ENABLE else str
    #
    # def yellow(self, str):
    #     return '\033[0;33m{}\033[0m'.format(str) if ENABLE else str
    #
    # def bgYellow(self, str):
    #     return '\033[0;30;43m{}\033[0m'.format(str) if ENABLE else str

    def green(self,str):
        print('\033[0;32m{}\033[0m'.format(str))

    def bgGreen(self,str):
        print('\033[0;30;42m{}\033[0m'.format(str))

    def red(self, str):
        print('\033[0;31m{}\033[0m'.format(str))

    def bgRed(self, str):
        print('\033[0;30;41m{}\033[0m'.format(str))

    def yellow(self, str):
        print('\033[0;33m{}\033[0m'.format(str))

    def bgYellow(self, str):
        print('\033[0;30;43m{}\033[0m'.format(str))


C = Color()
