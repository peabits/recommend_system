import os
import numpy as np
import pandas as pd
from datetime import datetime
from rec.env import project_dir
from rec.util import Print


def label_encode(_df):
    _set = set(_df)
    df_map = {y: x for x, y in enumerate(_set)}
    return _df.map(df_map), len(_set)


def encode_id(_id):
    return label_encode(_id)


def encode_gender(gender):
    gender_map = {1: 0, 2: 1}
    return gender.map(gender_map), 2


def encode_age(age, rule=None):
    def calc_age(birth):
        return datetime.now().year - birth

    def enc_age(_age):
        for i, v in enumerate(rule):
            if _age < v:
                return i
        else:
            return len(rule)

    if rule is None:
        rule = [18, 25, 30, 40]
    return age.apply(calc_age).apply(enc_age), len(rule)+1


def encode_skin(skin):
    return label_encode(skin)


def encode_name(name):
    return label_encode(name)


def encode_price(price):
    _max = int(price.max() // 10) + 1
    return price.map(lambda _price: int(_price // 10)), _max


def encode_category_1st(category):
    return label_encode(category)


def encode_category_2nd(category):
    return label_encode(category)


def encode_apply_age(age):
    return label_encode(age)


def encode_apply_part(part):
    return label_encode(part)


def encode_apply_skin(skin):
    return label_encode(skin)


def encode_member(member, member_embed_dict):
    member["member_id"], member_embed_dict["member_id"] = encode_id(member["member_id"])
    member["member_gender"], member_embed_dict["member_gender"] = encode_gender(member["member_gender"])
    member["member_age"], member_embed_dict["member_age"] = encode_age(member["member_age"])
    member["member_skin"], member_embed_dict["member_skin"] = encode_skin(member["member_skin"])
    return member


def encode_product(product, product_embed_dict):
    product["product_id"], product_embed_dict["product_id"] = \
        encode_id(product["product_id"])
    product["product_name"], product_embed_dict["product_name"] = \
        encode_name(product["product_name"])
    product["product_member_price"], product_embed_dict["product_member_price"] = \
        encode_price(product["product_member_price"])
    product["product_category_1st"], product_embed_dict["product_category_1st"] = \
        encode_category_1st(product["product_category_1st"])
    product["product_category_2nd"], product_embed_dict["product_category_2nd"] = \
        encode_category_2nd(product["product_category_2nd"])
    product["product_apply_age"], product_embed_dict["product_apply_age"] = \
        encode_apply_age(product["product_apply_age"])
    product["product_apply_part"], product_embed_dict["product_apply_part"] = \
        encode_apply_part(product["product_apply_part"])
    product["product_apply_skin"], product_embed_dict["product_apply_skin"] = \
        encode_apply_skin(product["product_apply_skin"])
    return product


def encode_record(record):
    record["member_id"], _ = encode_id(record["member_id"])
    record["product_id"], _ = encode_id(record["product_id"])
    return record


def get_member(member_filepath):
    df_member = pd.read_csv(filepath_or_buffer=member_filepath,
                            header=0,
                            names=["member_gender", "member_id", "member_age", "member_skin"],
                            usecols=["member_id", "member_gender", "member_age", "member_skin"],
                            encoding="utf-8")
    df_member = df_member[["member_id", "member_gender", "member_age", "member_skin"]]
    member_embed_dict = {}
    df_member = encode_member(df_member, member_embed_dict)
    return df_member, member_embed_dict


def get_product(product_filepath):
    df_product = pd.read_csv(filepath_or_buffer=product_filepath,
                             header=0,
                             names=["product_id", "product_name", "product_category_2nd", "product_category_1st",
                                    "product_apply_age", "product_apply_part", "product_apply_skin",
                                    "product_member_price"],
                             usecols=["product_id", "product_name", "product_member_price",
                                      "product_category_1st", "product_category_2nd",
                                      "product_apply_age", "product_apply_part", "product_apply_skin"],
                             encoding="utf-8")
    df_product = df_product[["product_id", "product_name", "product_member_price",
                             "product_category_1st", "product_category_2nd",
                             "product_apply_age", "product_apply_part", "product_apply_skin"]]
    product_embed_dict = {}
    df_product = encode_product(df_product, product_embed_dict)
    return df_product, product_embed_dict


def get_record(record_filepath):
    df_record = pd.read_csv(filepath_or_buffer=record_filepath,
                            header=0,
                            names=["member_id", "product_id", "amount", "amount_norm", "quantity", "quantity_norm"],
                            usecols=["member_id", "product_id", "amount_norm", "quantity_norm"],
                            dtype={"member_id": str, "product_id": str, "amount_norm": float, "quantity_norm": float},
                            encoding="utf-8")
    df_record = encode_record(df_record)
    return df_record


def load_data():
    member_filepath = f"{project_dir}/data/bj_member.csv"
    product_filepath = f"{project_dir}/data/complete_prd.csv"
    record_filepath = f"{project_dir}/data/bj_member_rate.csv"
    df_member, member_embed_dict = get_member(member_filepath)
    df_product, product_embed_dict = get_product(product_filepath)
    df_record = get_record(record_filepath)
    data = pd.merge(pd.merge(df_record, df_member), df_product)
    data = data[["member_id", "member_gender", "member_age", "member_skin",
                 "product_id", "product_name", "product_member_price",
                 "product_category_1st", "product_category_2nd",
                 "product_apply_age", "product_apply_part", "product_apply_skin",
                 # "amount_norm",
                 "quantity_norm"]]
    return data, member_embed_dict, product_embed_dict


if __name__ == '__main__':
    a, b, c = load_data()
    print(a.iloc[0, :])
