import jieba
import logging


def test1():
    jieba.setLogLevel(log_level=logging.INFO)
    string = "老头环这个游戏真好玩"
    seg_list = jieba.cut(string)
    print('/'.join(list(seg_list)))


if __name__ == '__main__':
    test1()
