import os
import pickle
from math import sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import wfdb as wf


def extract_data(data_path):
    global_data = []
    global_binary_labels = []
    paths = glob(os.path.join(data_path, '*.atr'))
    path_name_samples = [os.path.splitext(path)[0] for path in paths]
    for path_name in tqdm(path_name_samples):
        ann = wf.rdann(path_name, 'atr')
        record = wf.io.rdrecord(path_name)
        data = record.p_signal[:, 0]
        labels = ann.sample
        binary_labels = [0] * len(data)
        for label in labels:
            binary_labels[label] = 1
        global_data += data.tolist()
        global_binary_labels += binary_labels
    return global_data, global_binary_labels


def standardize(X):
    mean = sum(X) / len(X)
    deviation = 0
    for val in X:
        deviation += pow(val - mean, 2)
    stand = lambda value_i: (value_i - mean) / sqrt(deviation / len(X))
    return list(map(stand, X))


def normalize(X):
    min_data = min(X)
    max_data = max(X)
    norm = lambda f: (f - min_data) / (max_data - min_data)
    return list(map(norm, X))


def my_plot(X, y, y_pred):
    plt.plot(list(range(len(X))), X)
    index_labels = y
    labels = [x for x in range(len(index_labels)) if index_labels[x] == 1]
    plt.plot(labels, [0.4] * len(labels), 'o', color='green')
    index_labels = y_pred
    labels = [x for x in range(len(index_labels)) if index_labels[x] == 1]
    plt.plot(labels, [0.45] * len(labels), 'o', color='red')
    plt.show()


def proportion_of_ones(y):
    return y.count(1) / len(y)


def save_data(dataset_path):
    x1, y1 = extract_data(dataset_path)
    x1_final = normalize(x1)

    with open(f"X_data", "wb") as fd:
        pickle.dump(x1_final, fd)
    with open(f"y_labels", "wb") as fl:
        pickle.dump(y1, fl)


def transform_logits_for_metrics(logits, y):
    corrected_logits = logits.copy()
    for idx, label in enumerate(y):
        if label == 1:
            i = idx - 5 if idx >= 0 else 0
            while i <= idx + 5 and i <= len(y) - 1:
                if y[i] != 1:
                    corrected_logits[i] = 0
                i += 1
    return corrected_logits


def eval_precision(y, y_pred):
    tp = 0
    fp = 0
    for idx, label in enumerate(y_pred):
        if label == 1:
            i = idx - 5 if idx >= 5 else 0
            flag = False
            while i <= idx + 5 and i <= len(y) - 1:
                if y[i] == 1:
                    tp += 1
                    flag = True
                    break
                i+=1
            if not flag:
                fp += 1
    return 1 if tp+fp == 0 else tp / (tp+fp)


def eval_F1(precision, recall):
    return 0 if (precision + recall) ==0 else 2 * (precision * recall) / (precision + recall)

if __name__ == '__main__':
    dataset_path = os.path.join(os.getcwd(), "mit-bih-arrhythmia-database-1.0.0")
    save_data(dataset_path)