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
        self.init()

    def init(self):
        feature = pickle.load(open(f"{project_dir}/param/feature.pkl", "rb"))
        data = pickle.load(open(f"{project_dir}/param/data.pkl", "rb"))
        self.member_feature = feature["member_feature"]
        self.product_feature = feature["product_feature"]
        self.member_data = data["member_data"]
        self.product_data = data["product_data"]

    def getKnnItem(self, item_id, item_name="product", k=1):
        def getCosineSimilarity(vec1, vec2):
            cosine_sim = float(vec1.dot(vec2.T).item()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return cosine_sim

        assert k >= 1, "k >= 1"
        assert item_name in ["product", "member"], "item_name should be product or member"

        if item_name == "product":
            feature = self.product_feature
        else:
            feature = self.member_feature

        assert item_id in feature.keys(), "Expect item whose id is item id exists, but get None"

        feature_item = feature[item_id]

        res_set = [(_id, getCosineSimilarity(feature_item, ft)) for _id, ft in feature.items()]
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

    def load_id_map(self):
        _member_id_map = self.load_member_id_map()
        _product_id_map = self.load_product_id_map()
        return _member_id_map["ix2id"], _product_id_map["ix2id"]

    def load_member_id_map(self):
        return pickle.load(open(f"{project_dir}/param/member_id.pkl", "rb"))

    def load_product_id_map(self):
        return pickle.load(open(f"{project_dir}/param/product_id.pkl", "rb"))

    def p2p(self, p_ix2id=None):
        if p_ix2id is None:
            p_ix2id = self.load_product_id_map()["ix2id"]
        results = []
        for i, pix in enumerate(p_ix2id):
            pid = p_ix2id[pix]
            try:
                p10 = self.getKnnItem(pix, k=10)
            except AssertionError as ae:
                continue
            result = [pid]
            for p in p10:
                result.append(p_ix2id[p])
            results.append(result)
            print(f"p2p -> {round(i/len(p_ix2id), 2)}")
        columns = ["product_id"]
        columns.extend([f"{i+1}" for i in range(10)])
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv(path_or_buf=f"{project_dir}/result/p2p.csv", 
                  index=False, 
                  columns=columns)

    def m2p(self, m_ix2id=None, p_ix2id=None):
        if m_ix2id is None:
            m_ix2id = self.load_member_id_map()["ix2id"]
        if p_ix2id is None:
            p_ix2id = self.load_product_id_map()["ix2id"]
        results = []
        for i, mix in enumerate(m_ix2id):
            mid = m_ix2id[mix]
            try:
                p10 = self.getMemberMostLike(mix, k=10)
            except AssertionError as ae:
                continue
            result = [mid]
            for p in p10:
                result.append(p_ix2id[p])
            results.append(result)
            print(f"m2p -> {round(i/len(m_ix2id), 2)}")
        columns = ["member_id"]
        columns.extend([f"{i+1}" for i in range(10)])
        df = pd.DataFrame(data=results, columns=columns)
        df.to_csv(path_or_buf=f"{project_dir}/result/m2p.csv", 
                  index=False, 
                  columns=columns)

    def m2m2p(self, m_ix2id=None, p_ix2id=None):
        if m_ix2id is None:
            m_ix2id = self.load_member_id_map()["ix2id"]
        if p_ix2id is None:
            p_ix2id = self.load_product_id_map()["ix2id"]
        m2p_dict = {}
        for i, mix in enumerate(m_ix2id):
            mid = m_ix2id[mix]
            try:
                m2p_dict[mix] = self.getMemberMostLike(mix)
            except:
                continue
            print(f"m2m2p.1 -> {i} / {len(m_ix2id)}")
        results = []
        for i, mix in enumerate(m_ix2id):
            mid = m_ix2id[mix]
            try:
                m10 = self.getKnnItem(mix, "member", k=10)
            except AssertionError as ae:
                continue
            result = [mid]
            for m in m10:
                result.append(p_ix2id[m2p_dict[m]])
            results.append(result)
            print(f"m2m2p.2 -> {i} / {len(m_ix2id)}")
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
