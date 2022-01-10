import os

import torch
import pytorch_lightning as pl
from DataModule import ECGdataModule
from LSTM import LSTM
from pytorch_lightning import seed_everything

if __name__ == '__main__':
    seed_everything(100)
    data_path = os.path.join(os.getcwd(), "X_data")
    labels_path = os.path.join(os.getcwd(), "y_labels")
    p = {"batch_size": 32,
         "hidden_size": 64,
         "learning_rate": 0.004,
         "seq_len": 80,
         "weight": 0.00361,
         "max_epochs": 10
         }

    data_module = ECGdataModule(data_path, labels_path, batch_size=p["batch_size"], seq_len=p["seq_len"])

    loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([p["weight"], 1 - p["weight"]]))
    model = LSTM(input_size=1, hidden_size=p["hidden_size"], seq_len=p["seq_len"], batch_size=p["batch_size"],
                 num_layers=2, learning_rate=p["learning_rate"], dropout=0.2, criterion=loss)

    trainer = pl.Trainer(limit_train_batches=0.2, max_epochs=p["max_epochs"])

    hyperparameters = dict(learning_rate=p["learning_rate"], weight=p["weight"], batch_size=p["batch_size"],
                           seq_len=p["seq_len"], hidden_size=p["hidden_size"], num_layers=2, dropout=0.2
                           )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
