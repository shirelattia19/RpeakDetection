import os
import pickle
import pytest
from torch.utils.data import DataLoader
from utils import eval_precision
from DataModule import ECGdataSet


# @pytest.mark.parametrize('data_set', [data_set_arrhythmia_binary()])
# def test_binary_labels(data_set):
#     for i in range(len(data_set)):
#         data, binary_labels = data_set[i]
#
#         assert len(data) == len(binary_labels)
#         # for label in data_set.labels[i]:
#         #     assert binary_labels[label] == 1
#
#
# @pytest.mark.parametrize('data_set', [data_set_arrhythmia_binary()])
# def test_loader(data_set):
#     loader = DataLoader(data_set, batch_size=2)
#     for idx, batch in enumerate(loader):
#         assert batch

@pytest.fixture()
def X_data():
    data_path = os.path.join(os.getcwd(), "..", "X_data")
    with open(data_path, "rb") as fd:
        X = pickle.load(fd)
    return X


@pytest.fixture()
def y_labels():
    labels_path = os.path.join(os.getcwd(), "..", "y_labels")
    with open(labels_path, "rb") as fl:
        y = pickle.load(fl)
    return y


def test_loader(X_data, y_labels):
    data_set = ECGdataSet(X_data, y_labels, seq_len=3000)
    loader = DataLoader(data_set, batch_size=32)
    for batch in loader:
        assert batch


def test_binary_labels(X_data, y_labels):
    data_set = ECGdataSet(X_data, y_labels, seq_len=3000)
    for i in range(len(data_set)):
        data, binary_labels = data_set[i]
        assert len(data) == len(binary_labels)


def test_show_plot():
    data_set = ECGdataSet(X_data, y_labels, seq_len=3000)
    data_set.show_plot(40)


def test_eval_precision():
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    logits = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    res = eval_precision(y, logits)
    assert res == 1


def test_eval_precision2():
    y = [0, 1, 0, 0, 0, 0, 0, 0, 0]
    logits = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    res = eval_precision(y, logits)
    assert res == 1


def test_eval_precision3():
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    logits = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    res = eval_precision(y, logits)
    assert res == 1


def test_eval_precision4():
    y = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    logits = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    res = eval_precision(y, logits)
    assert res == 1


def test_eval_precision5():
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    logits = [1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    res = eval_precision(y, logits)
    assert res == 11 / 12
