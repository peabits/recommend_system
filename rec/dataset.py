from abc import ABC

from torch.utils.data import Dataset
from rec.preprocess import load_data

import pickle
import pandas as pd
import torch


class RecDataset(Dataset, ABC):

    def __init__(self, _data):
        super().__init__()
        self.data = _data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        member_id = self.data.iloc[i, 0]
        member_gender = self.data.iloc[i, 1]
        member_age = self.data.iloc[i, 2]
        member_skin = self.data.iloc[i, 3]
        product_id = self.data.iloc[i, 4]
        product_name = self.data.iloc[i, 5]
        product_member_price = self.data.iloc[i, 6]
        product_category_1st = self.data.iloc[i, 7]
        product_category_2nd = self.data.iloc[i, 8]
        product_apply_age = self.data.iloc[i, 9]
        product_apply_part = self.data.iloc[i, 10]
        product_apply_skin = self.data.iloc[i, 11]
        target = torch.FloatTensor([self.data.iloc[i, 12]])
        member_inputs = {
            "member_id": torch.LongTensor([member_id]).view(1, -1),
            "member_gender": torch.LongTensor([member_gender]).view(1, -1),
            "member_age": torch.LongTensor([member_age]).view(1, -1),
            "member_skin": torch.LongTensor([member_skin]).view(1, -1)
        }
        product_inputs = {
            "product_id": torch.LongTensor([product_id]).view(1, -1),
            "product_name": torch.LongTensor([product_name]).view(1, -1),
            "product_member_price": torch.LongTensor([product_member_price]).view(1, -1),
            "product_category_1st": torch.LongTensor([product_category_1st]).view(1, -1),
            "product_category_2nd": torch.LongTensor([product_category_2nd]).view(1, -1),
            "product_apply_age": torch.LongTensor([product_apply_age]).view(1, -1),
            "product_apply_part": torch.LongTensor([product_apply_part]).view(1, -1),
            "product_apply_skin": torch.LongTensor([product_apply_skin]).view(1, -1)
        }
        item = {
            "member_inputs": member_inputs,
            "product_inputs": product_inputs,
            "target": target
        }
        return item
