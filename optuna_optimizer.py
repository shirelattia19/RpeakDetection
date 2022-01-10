import argparse
import os

import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (plot_intermediate_values, plot_optimization_history, plot_param_importances)
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from DataModule import ECGdataModule
from LSTM import LSTM

MAX_EPOCHS = 10
SEED_VAL = 42
LOGS_PATH = os.path.join(os.getcwd(),"lightning_logs")
MODEL_DIR = os.path.join(os.getcwd(),"trials")
data_path = os.path.join(os.getcwd(), "X_data")
labels_path = os.path.join(os.getcwd(), "y_labels")


def objective(trial: optuna.trial.Trial) -> float:
    batch_size = trial.suggest_int("batch_size", 16, 128)
    seq_len = trial.suggest_int("seq_len", 10, 150)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.0008, 0.006)
    max_epochs = MAX_EPOCHS
    hidden_size = trial.suggest_int("hidden_size", 32, 400)
    num_layers = trial.suggest_int('num_layers', 2, 3)
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    weight = 0.00361
    #checkpoint_filename = f'trial_{trial.number}'
    CHECKPOINTS_PATH = os.path.join(MODEL_DIR, "trial_{}".format(trial.number))

    logger = pl_loggers.TensorBoardLogger(save_dir=LOGS_PATH)

    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger,
                         callbacks=[PyTorchLightningPruningCallback(trial, monitor="valid/loss")],
                         default_root_dir=CHECKPOINTS_PATH, limit_train_batches=0.1)

    data_module = ECGdataModule(data_path, labels_path, batch_size=batch_size, seq_len=seq_len)

    loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([weight, 1 - weight]))
    model = LSTM(input_size=1, hidden_size=hidden_size, seq_len=seq_len, batch_size=batch_size, num_layers=num_layers,
                 learning_rate=learning_rate, dropout=dropout, criterion=loss)

    hyperparameters = dict(learning_rate=learning_rate, weight=weight, batch_size=batch_size,
                           seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
                           )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=data_module)

    curr_loss = trainer.callback_metrics['valid/loss'].item()

    return curr_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="minimize"
                                , pruner=pruner)
    study.optimize(objective, n_trials=20)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  Full Summary of Trials:  ")
    print(study.trials_dataframe())

    plot_optimization_history(study).show()
    plot_intermediate_values(study).show()
    try:
        plot_param_importances(study).show()
    except ValueError:
        pass
