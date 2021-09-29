import numpy as np
import pickle
from rec.env import project_dir
import pandas as pd


class RecInterface(object):
    def __init__(self):
        super().__init__()
        self.member_feature = None
        self.product_feature = None
        self.member_data = None
        self.product_data = None
        self.member_ix2id = None
        self.product_ix2id = None
        self.init()

    def init(self):
        feature = pickle.load(open(f"{project_dir}/param/feature.pkl", "rb"))
        data = pickle.load(open(f"{project_dir}/param/data.pkl", "rb"))
        ix2id = pickle.load(open(f"{project_dir}/param/ix2id.pkl", "rb"))
        self.member_feature = feature["member_feature"]
        self.product_feature = feature["product_feature"]
        self.member_data = data["member_data"]
        self.product_data = data["product_data"]
        self.member_ix2id = ix2id["member_ix2id"]
        self.product_ix2id = ix2id["product_ix2id"]

    def getCosineSimilarity(self, vec1, vec2):
        cosine_sim = float(vec1.dot(vec2.T).item()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_sim   

    def getKnnProduct(self, product_id, k=1):
        assert k >= 1, "k >= 1"
        product_feature = self.product_feature
        assert product_id in product_feature.keys(), "Expect item whose id is item id exists, but get None"
        product_feature_item = product_feature[product_id]
        res_set = [(_id, self.getCosineSimilarity(product_feature_item, ft)) for _id, ft in product_feature.items()]
        res_set = sorted(res_set, key=lambda x: x[1], reverse=True)
        return [res_set[i][0] for i in range(k + 1)][1:]

    def getKnnMember(self, member_id,  k=1):
        assert k >= 1, "k >= 1"
        member_feature = self.member_feature
        assert member_id in member_feature.keys(), "Expect item whose id is item id exists, but get None"
        member_feature_item = member_feature[member_id]
        res_set = [(_id, self.getCosineSimilarity(member_feature_item, ft)) for _id, ft in member_feature.items()]
        res_set = sorted(res_set, key=lambda x: x[1], reverse=True)
        return [res_set[i][0] for i in range(k + 1)][1:]

    def getMemberMostLike(self, member_id, k=1):
        assert member_id in self.member_feature, 'Expect user whose id is member id exists, but get None'
        feature = self.member_feature[member_id]
        product = self.product_data
        rank = {}
        for product_id in product.keys():
            feature_product = self.product_feature[product_id]
            _rank = np.dot(feature, feature_product.T)
            if product_id not in rank.keys():
                rank[product_id] = _rank.item()
        rank = [(i, r) for i, r in rank.items()]
        res_set = [_id[0] for _id in sorted(rank, key=lambda x: x[1], reverse=True)]
        return res_set[0:k] if k > 1 else res_set[0]

    def p2p(self, p_ix2id=None):
        if p_ix2id is None:
            p_ix2id = self.product_ix2id
        results = []
        for i, pix in enumerate(p_ix2id):
            pid = p_ix2id[pix]
            try:
                pix10 = self.getKnnProduct(pix, k=10)
            except AssertionError as ae:
                continue
            result = [pid]
            for pix in pix10:
                pid = p_ix2id[pix]
                result.append(pid)
            results.append(result)
            print(f"p2p -> {i} / {len(p_ix2id)}")
        columns = ["product_id"]
        columns.extend([f"{i+1}" for i in range(10)])
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv(path_or_buf=f"{project_dir}/result/p2p.csv", 
                  index=False, 
                  columns=columns)

    def m2m(self, m_ix2id=None):
        if m_ix2id is None:
            m_ix2id = self.member_ix2id
        results = []
        for i, mix in enumerate(m_ix2id):
            mid = m_ix2id[mix]
            try:
                m10 = self.getKnnMember(mix, k=10)
            except AssertionError as ae:
                continue
            result = [mid]
            for p in m10:
                result.append(m_ix2id[p])
            results.append(result)
            print(f"m2m -> {i} / {len(m_ix2id)}")
        columns = ["member_id"]
        columns.extend([f"{i+1}" for i in range(10)])
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv(path_or_buf=f"{project_dir}/result/m2m.csv", 
                  index=False, 
                  columns=columns)

    def m2p(self, m_ix2id=None, p_ix2id=None):
        if m_ix2id is None:
            m_ix2id = self.member_ix2id
        if p_ix2id is None:
            p_ix2id = self.product_ix2id
        results = []
        for i, mix in enumerate(m_ix2id):
            mid = m_ix2id[mix]
            try:
                pix10 = self.getMemberMostLike(mix, k=10)
            except AssertionError as ae:
                continue
            result = [mid]
            for pix in pix10:
                result.append(pix)
            results.append(result)
            print(f"m2p -> {i} / {len(m_ix2id)}")
        columns = ["member_id"]
        columns.extend([f"{i+1}" for i in range(10)])
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv(path_or_buf=f"{project_dir}/result/m2p.csv", 
                  index=False, 
                  columns=columns)

    def m2p2p(self, m_ix2id=None, p_ix2id=None):
        if m_ix2id is None:
            m_ix2id = self.member_ix2id
        if p_ix2id is None:
            p_ix2id = self.product_ix2id
        results = []
        for i, mix in enumerate(m_ix2id):
            mid = m_ix2id[mix]
            try:
                pix = self.getMemberMostLike(mix)
                pix10 = self.getKnnProduct(pix, k=10)
            except AssertionError as ae:
                continue
            result = [mid]
            for pix in pix10:
                pid = p_ix2id[pix]
                result.append(pid)
            results.append(result)
            print(f"m2p2p -> {i} / {len(m_ix2id)}")
        columns = ["member_id"]
        columns.extend([f"{i+1}" for i in range(10)])
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv(path_or_buf=f"{project_dir}/result/m2p2p.csv",
                  index=False,
                  columns=columns)

    def m2m2p(self, m_ix2id=None, p_ix2id=None):
        if m_ix2id is None:
            m_ix2id = self.member_ix2id
        if p_ix2id is None:
            p_ix2id = self.product_ix2id
        results = []
        for i, mix in enumerate(m_ix2id):
            mid = m_ix2id[mix]
            try:
                mix10 = self.getKnnMember(mix, k=10)
            except AssertionError as ae:
                continue
            result = [mid]
            for mix in mix10:
                pix = self.getMemberMostLike(mix)
                pid = p_ix2id[pix]
                result.append(pid)
            results.append(result)
            print(f"m2m2p -> {i} / {len(m_ix2id)}")
        columns = ["member_id"]
        columns.extend([f"{i+1}" for i in range(10)])
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv(path_or_buf=f"{project_dir}/result/m2m2p.csv",
                  index=False,
                  columns=columns)


def main():
    feature = pickle.load(open(f"{project_dir}/param/feature.pkl", "rb"))
    print(feature)


if __name__ == '__main__':
    main()
