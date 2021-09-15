import numpy as np
import pickle


def getKnnItem(item_id, item_name="product", k=1):
    assert k >= 1

    def getCosineSimilarity(vec1, vec2):
        cosine_sim = float(vec1.dot(vec2.T).item()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_sim

    feature = pickle.load(open("param/feature.pkl", "rb"))
    feature = feature[f"{item_name}_feature"]
    assert item_id in feature.keys()
    feature_item = feature[item_id]

    res_set = [(_id, getCosineSimilarity(feature_item, ft)) for _id, ft in feature.items()]
    res_set = sorted(res_set, key=lambda x: x[1], reverse=True)

    return [res_set[i][0] for i in range(k+1)][1:]


def getMemberMostLike(member_id, k=1):
    feature = pickle.load(open("param/feature.pkl", "rb"))
    data = pickle.load(open("param/data.pkl", "rb"))
    assert member_id in data['member'], \
        'Expect user whose id is uid exists, but get None'
    feature_member = feature['member_feature'][member_id]

    product = data['product']
    rank = {}
    for product_id in product.keys():
        feature_product = feature['product_feature'][product_id]
        _rank = np.dot(feature_member, feature_product.T)
        if product_id not in rank.keys():
            rank[product_id] = _rank.item()

    rank = [(i, r) for i, r in rank.items()]
    res_set = [_id[0] for _id in sorted(rank, key=lambda x: x[1], reverse=True)]

    return res_set[0:k]
