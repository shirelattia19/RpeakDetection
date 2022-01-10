import torch
import pytorch_lightning as pl
import torchmetrics
from torch import tensor
from torch.nn import functional as F
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score


#
# def get_weight_tensor(y):
#     res = []
#     for elem in y:
#         if elem == 1:
#             res.append(0.0038)
#         else:
#             res.append(1 - 0.0038)
#     return torch.Tensor(res)
from utils import my_plot, eval_precision, eval_F1


class LSTM(pl.LightningModule):

    def __init__(self, input_size, hidden_size, seq_len, batch_size, num_layers, dropout, learning_rate, criterion):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.train_acc = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()

        self.valid_acc = torchmetrics.Accuracy()
        self.valid_precision = torchmetrics.Precision()
        self.test_acc = torchmetrics.Accuracy()
        self.test_precision = torchmetrics.Precision()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  dropout=dropout, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=2)
        self.counter =0

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_hat = self.linear(lstm_out)#[:, -1])
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y_pred = F.softmax(y_pred, dim=2)
        loss = self.criterion(y_pred.view(-1, 2), y.view(-1))
        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        logits = torch.argmax(y_pred, dim=2)
        self._log_metrics(y, logits, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.counter+=1
        x, y = batch
        y_pred = self.forward(x)
        y_pred: tensor = F.softmax(y_pred, dim=2)
        loss = self.criterion(y_pred.view(-1, 2), y.view(-1))
        self.log('valid/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        logits = torch.argmax(y_pred, dim=2)
        self._log_metrics(y, logits, "valid")
        if self.counter % 8000:
            my_plot(x[-1], y[-1], logits[-1])
        return loss

    def test_step(self, batch, batch_idx):
        self.counter += 1
        x, y = batch
        y_pred = self.forward(x)
        y_pred = F.softmax(y_pred, dim=2)
        loss = self.criterion(y_pred.view(-1, 2), y.view(-1))
        self.log('test/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        logits = torch.argmax(y_pred, dim=2)
        self._log_metrics(y, logits, "test")
        return loss

    def _log_metrics(self, y, logits, name):
        acc = accuracy_score(y.flatten(), logits.flatten())
        self.log(f'{name}/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        precision = eval_precision(y.flatten(), logits.flatten())
        self.log(f'{name}/precision', precision, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        recall = recall_score(y.flatten(), logits.flatten(), zero_division=1)
        self.log(f'{name}/recall', recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        f1 = eval_F1(precision, recall)
        self.log(f'{name}/f1', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
