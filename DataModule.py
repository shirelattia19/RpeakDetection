import pickle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl


class ECGdataSet(Dataset):

    def __init__(self, X, y, seq_len: int = 1):
        super(ECGdataSet).__init__()
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len - 1)

    # def __getitem__(self, index):
    #     return torch.unsqueeze(self.X[index:index + self.seq_len], dim=1), \
    #            self.y[index + self.seq_len - 1]
    def __getitem__(self, index):
        return torch.unsqueeze(self.X[index], dim=1), \
               torch.unsqueeze(self.y[index], dim=1)

    def show_plot(self, index, seq_len=None):
        if not seq_len:
            seq_len = self.seq_len
        plt.plot(list(range(seq_len)), self.X[index:index + seq_len])
        index_labels = self.y[index:index + seq_len]
        labels = [x for x in range(len(index_labels)) if index_labels[x] == 1]
        plt.plot(labels, [2] * len(labels), 'o', color='red')
        plt.show()


class ECGdataModule(pl.LightningDataModule):

    def __init__(self, data_path, labels_path, batch_size: int, seq_len: int = 1, num_workers=2):
        super(ECGdataModule).__init__()
        self.data_path = data_path
        self.labels_path = labels_path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def prepare_data(self):
        with open(self.data_path, "rb") as fd:
            self.X = pickle.load(fd)
        with open(self.labels_path, "rb") as fl:
            self.y = pickle.load(fl)

    def setup(self, stage=None):
        sequenced_data = []
        sequenced_labels = []
        for i in range((len(self.X) + self.seq_len - 1) // self.seq_len):
            left_boundery = i * self.seq_len
            right_boundery = (i + 1) * self.seq_len
            sequenced_data.append(self.X[left_boundery:right_boundery])
            sequenced_labels.append(self.y[left_boundery:right_boundery])

        self.X_train = sequenced_data[0:int(0.8 * len(sequenced_data))]
        self.y_train = sequenced_labels[0:int(0.8 * len(sequenced_data))]
        self.X_val = sequenced_data[int(0.8 * len(sequenced_data)):int(0.9 * len(sequenced_data))]
        self.y_val = sequenced_labels[int(0.8 * len(sequenced_data)):int(0.9 * len(sequenced_data))]
        self.X_test = sequenced_data[int(0.9 * len(sequenced_data)):]
        self.y_test = sequenced_labels[int(0.9 * len(sequenced_data)):]

    def train_dataloader(self):
        train_dataset = ECGdataSet(self.X_train, self.y_train, seq_len=self.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = ECGdataSet(self.X_val, self.y_val, seq_len=self.seq_len)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_dataset = ECGdataSet(self.X_test, self.y_test, seq_len=self.seq_len)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return test_loader
