import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from rec import dataset
import pickle


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


class RecModel(nn.Module):
    def __init__(self, member_embed_dict, product_embed_dict, data=None, embed_dim=32, fc_size=200):
        super().__init__()
        if data is not None:
            self.dataset = dataset.RecDataset(data)
        self.embedding_member_id = nn.Embedding(member_embed_dict["member_id"], embed_dim)
        self.embedding_member_gender = nn.Embedding(member_embed_dict["member_gender"], embed_dim // 2)
        self.embedding_member_age = nn.Embedding(member_embed_dict["member_age"], embed_dim // 2)
        self.embedding_member_skin = nn.Embedding(member_embed_dict["member_skin"], embed_dim // 2)

        self.fc_member_id = nn.Linear(embed_dim, embed_dim)
        self.fc_member_gender = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_member_age = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_member_skin = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_member = nn.Linear(4 * embed_dim, fc_size)

        self.embedding_product_id = nn.Embedding(product_embed_dict["product_id"], embed_dim)
        self.embedding_product_name = nn.Embedding(product_embed_dict["product_name"], embed_dim // 2)
        self.embedding_product_member_price = nn.Embedding(product_embed_dict["product_member_price"], embed_dim // 2)
        self.embedding_product_category_1st = nn.Embedding(product_embed_dict["product_category_1st"], embed_dim // 2)
        self.embedding_product_category_2nd = nn.Embedding(product_embed_dict["product_category_2nd"], embed_dim // 2)
        self.embedding_product_apply_age = nn.Embedding(product_embed_dict["product_member_price"], embed_dim // 2)
        self.embedding_product_apply_part = nn.Embedding(product_embed_dict["product_apply_part"], embed_dim // 2)
        self.embedding_product_apply_skin = nn.Embedding(product_embed_dict["product_apply_skin"], embed_dim // 2)

        self.fc_product_id = nn.Linear(embed_dim, embed_dim)
        self.fc_product_name = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_product_member_price = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_product_category_1st = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_product_category_2nd = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_product_apply_age = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_product_apply_part = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_product_apply_skin = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_product = nn.Linear(embed_dim * 8, fc_size)

        self.BatchNorm = nn.BatchNorm2d(1)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        self.MSELoss = nn.MSELoss()

    def forward(self, member_inputs, product_inputs):
        member_id = member_inputs["member_id"]
        member_gender = member_inputs["member_gender"]
        member_age = member_inputs["member_age"]
        member_skin = member_inputs["member_skin"]

        feature_member_id = self.BatchNorm(self.ReLU(
            self.fc_member_id(self.embedding_member_id(member_id))))
        feature_member_gender = self.BatchNorm(self.ReLU(
            self.fc_member_gender(self.embedding_member_gender(member_gender))))
        feature_member_age = self.BatchNorm(self.ReLU(
            self.fc_member_age(self.embedding_member_age(member_age))))
        feature_member_skin = self.BatchNorm(self.ReLU(
            self.fc_member_skin(self.embedding_member_skin(member_skin))))

        feature_member = self.Tanh(self.fc_member(
            torch.cat(tensors=[feature_member_id, feature_member_gender, feature_member_age, feature_member_skin],
                      dim=3)).view(-1, 1, 200))

        product_id = product_inputs["product_id"]
        product_name = product_inputs["product_name"]
        product_member_price = product_inputs["product_member_price"]
        product_category_1st = product_inputs["product_category_1st"]
        product_category_2nd = product_inputs["product_category_2nd"]
        product_apply_age = product_inputs["product_apply_age"]
        product_apply_part = product_inputs["product_apply_part"]
        product_apply_skin = product_inputs["product_apply_skin"]

        feature_product_id = self.BatchNorm(self.ReLU(
            self.fc_product_id(self.embedding_product_id(product_id))))
        feature_product_name = self.BatchNorm(self.ReLU(
            self.fc_product_name(self.embedding_product_name(product_name))))
        feature_product_member_price = self.BatchNorm(self.ReLU(
            self.fc_product_member_price(self.embedding_product_member_price(product_member_price))))
        feature_product_category_1st = self.BatchNorm(self.ReLU(
            self.fc_product_category_1st(self.embedding_product_category_1st(product_category_1st))))
        feature_product_category_2nd = self.BatchNorm(self.ReLU(
            self.fc_product_category_2nd(self.embedding_product_category_2nd(product_category_2nd))))
        feature_product_apply_age = self.BatchNorm(self.ReLU(
            self.fc_product_apply_age(self.embedding_product_apply_age(product_apply_age))))
        feature_product_apply_part = self.BatchNorm(self.ReLU(
            self.fc_product_apply_part(self.embedding_product_apply_part(product_apply_part))))
        feature_product_apply_skin = self.BatchNorm(self.ReLU(
            self.fc_product_apply_skin(self.embedding_product_apply_skin(product_apply_skin))))

        feature_product = self.Tanh(self.fc_product(
            torch.cat(tensors=[feature_product_id, feature_product_name, feature_product_member_price,
                               feature_product_category_1st, feature_product_category_2nd,
                               feature_product_apply_age, feature_product_apply_part, feature_product_apply_skin],
                      dim=3)).view(-1, 1, 200))

        output = torch.sum(feature_member * feature_product, 2)
        return output, feature_member, feature_product

    def fit(self, data):
        self.dataset = dataset.RecDataset(data)

    def train(self, epochs=5, lr=0.0001, mode=True):
        optimizer = optim.Adam(params=self.parameters(), lr=lr)

        dataloader = DataLoader(self.dataset, batch_size=256, shuffle=True)
        _len = len(dataloader)
        loss_lst = []
        for epoch in range(epochs):
            loss_lst.append([])
            loss_sum = 0
            for i, batch in enumerate(dataloader):
                member_inputs = batch["member_inputs"]
                product_inputs = batch["product_inputs"]
                target = batch["target"]
                self.zero_grad()
                predict, _, _ = self(member_inputs, product_inputs)
                loss = self.MSELoss(predict, target)
                loss_sum += loss
                loss.backward()
                optimizer.step()
                loss_lst[epoch].append(loss.detach().numpy())
                percent = round(i / _len * 100, ndigits=2)
                prefix = f"Epoch={epoch + 1} loss={loss.detach().numpy():<20}"
                progress(prefix, percent)
            print()

    def save(self, save_file="param/model.pkl"):
        torch.save(self.state_dict(), save_file)

    def load(self, load_file="param/model.pkl"):
        self.load_state_dict(torch.load(load_file))

    def save_feature(self):
        dataloader = DataLoader(self.dataset, batch_size=256, shuffle=True, num_workers=4)
        member_feature = {}
        product_feature = {}
        member = {}
        product = {}
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                member_inputs = batch["member_inputs"]
                product_inputs = batch["product_inputs"]

                _, feature_member, feature_product = self.__call__(member_inputs, product_inputs)
                # feature_member = feature_member.tolist()
                # feature_product = feature_product.tolist()

                feature_member = feature_member.numpy()
                feature_product = feature_product.numpy()

                for i in range(member_inputs["member_id"].shape[0]):
                    member_id = member_inputs["member_id"][i]
                    member_gender = member_inputs["member_gender"][i]
                    member_age = member_inputs["member_age"][i]
                    member_skin = member_inputs["member_skin"][i]

                    product_id = product_inputs["product_id"][i]
                    product_name = product_inputs["product_name"][i]
                    product_member_price = product_inputs["product_member_price"][i]
                    product_category_1st = product_inputs["product_category_1st"][i]
                    product_category_2nd = product_inputs["product_category_2nd"][i]
                    product_apply_age = product_inputs["product_apply_age"][i]
                    product_apply_part = product_inputs["product_apply_part"][i]
                    product_apply_skin = product_inputs["product_apply_skin"][i]

                    if member_id not in member.keys():
                        member[member_id.item()] = {
                            "member_id": member_id,
                            "member_gender": member_gender,
                            "member_age": member_age,
                            "member_skin": member_skin
                        }
                    if product_id not in product.keys():
                        product[product_id.item()] = {
                            "product_id": product_id,
                            "product_name": product_name,
                            "product_member_price": product_member_price,
                            "product_category_1st": product_category_1st,
                            "product_category_2nd": product_category_2nd,
                            "product_apply_age": product_apply_age,
                            "product_apply_part": product_apply_part,
                            "product_apply_skin": product_apply_skin
                        }

                    if member_id not in member_feature.keys():
                        member_feature[member_id.item()] = feature_member[i]
                    if product_id not in product_feature.keys():
                        product_feature[product_id.item()] = feature_product[i]

                print(f'[{idx:4}/{len(dataloader)}] Solved: {(idx + 1) * 256:} samples')

        feature = {"member_feature": member_feature, "product_feature": product_feature}
        data = {"member": member, "product": product}

        pickle.dump(feature, open("param/feature.pkl", "wb"))

        pickle.dump(data, open("param/data.pkl", "wb"))
