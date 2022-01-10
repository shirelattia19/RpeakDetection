import os
import torch
from pytorch_lightning import Trainer

from DataModule import ECGdataModule
from LSTM import LSTM

model_path = os.path.join(os.getcwd(), "model/epoch=9-step=19489.ckpt")
data_path = os.path.join(os.getcwd(), "X_data")
labels_path = os.path.join(os.getcwd(), "y_labels")
hparams_path = os.path.join(os.getcwd(), "model/hparams.yaml")



if __name__ == '__main__':
    batch_size = 32
    seq_len = 80
    weight = 0.0036
    loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([weight, 1 - weight]))
    model = LSTM.load_from_checkpoint(model_path, hparams_file=hparams_path, input_size=1, criterion=loss)
    model.eval()
    data_module = ECGdataModule(data_path, labels_path, batch_size=batch_size, seq_len=seq_len)
    trainer = Trainer()
    trainer.test(model, data_module)
