import json
import torch

class DataSplit():
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx: int):
        return self.data[idx]

class SingleDataset():
    def __init__(self, data_path: str = 'data/routines_1028/sample.json', classes_path: str = 'data/routines_1028/classes.json', test_perc = 0.2):
        self._alldata = read_data(data_path = data_path, classes_path = classes_path)
        print(len(self._alldata),' examples found in dataset.')
        edges, nodes, context_curr, context_query, y = self._alldata[0]
        self.n_nodes = edges.size()[1]
        self.n_len = nodes.size()[1]
        self.e_len = edges.size()[-1]
        self.c_len = context_curr.size()[0]
        random.shuffle(self._alldata)
        num_test = int(round(test_perc*len(self._alldata)))
        self.test = DataSplit(self._alldata[:num_test])
        self.train = DataSplit(self._alldata[num_test:])
        print(len(self.train),' examples in train split.')
        print(len(self.test),' examples in test split.')
        losses = torch.Tensor([0])
        for d in self.train:
            r = torch.randn(self.n_nodes, self.n_nodes, self.e_len)
            random_out = torch.nn.Sigmoid()(r)
            losses = losses + torch.nn.BCELoss()(random_out, d[4])
        print("random loss : ", losses/len(self.train))

    def get_train_loader(self):
        return torch.utils.data.DataLoader(self.train, num_workers=8, batch_size=10)

    def get_test_loader(self):
        return torch.utils.data.DataLoader(self.test, num_workers=8, batch_size=10)



with open('data/routines_1028/classes.json', 'r') as f:
    classes = json.load(f)
