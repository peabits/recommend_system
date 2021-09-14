from rec import dataset
from rec import preprocess
from rec import model
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import pickle
import tqdm
import torch


def progress(prefix, percent, width=50):
    """
    进度打印功能
    :param percent: 进度
    :param width: 进度条长度
    """
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print(f'\r{prefix} {show_str} {percent:<5}%', end='')


def main():
    data, member_embed_dict, product_embed_dict = preprocess.load_data()
    rec_model = model.RecModel(member_embed_dict, product_embed_dict)
    rec_model.fit(data)
    rec_model.train()
    torch.save(rec_model.state_dict(), "param/model.pkl")


if __name__ == '__main__':
    main()
