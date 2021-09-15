from rec import dataset
from rec import interface
from rec import model
from rec import preprocess
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import pickle
import tqdm
import torch


def progress(prefix, percent, width=50):
    """
    进度打印功能
    :param prefix:
    :param percent: 进度
    :param width: 进度条长度
    """
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print(f'\r{prefix} {show_str} {percent:<5}%', end='')


def main():
    # 加载数据集
    # data, member_embed_dict, product_embed_dict = preprocess.load_data()

    # 模型
    # rec_model = model.RecModel(member_embed_dict, product_embed_dict, data=data)
    # rec_model.fit(data)
    # rec_model.train()
    # torch.save(rec_model.state_dict(), "param/model.pkl")

    # rec_model.load()
    # rec_model.save_feature()

    # 测试
    print(interface.getKnnItem(2, k=10))
    print(interface.getKnnItem(item_id=2, item_name="member", k=10))
    print(interface.getMemberMostLike(2, k=10))


if __name__ == '__main__':
    main()
