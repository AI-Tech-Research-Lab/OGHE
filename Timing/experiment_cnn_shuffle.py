import argparse
import pickle
import lightning as L
import logging
import numpy as np
import structlog
from itertools import product
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

from data_utils import create_stratified_folds


log = structlog.get_logger()

################ WARNING
# In this notebook, we are disactivating the bias in the ConvLayers.
# We do this to avoid this bug of TenSEAL: https://github.com/OpenMined/TenSEAL/issues/240
# that makes the encrypted conv1d return a wrong result.
# It does not affect the validity of the solution in any way: the bias could be used by just fixing that bug in TenSEAL
# or using another method to pack the ciphertexts in the flatten operation.

CLASS_NAMES = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
CLASS_WEIGHTS = [0.8179, 1.0483, 0.3300, 1.4160, 0.8214, 0.9632, 1.4160, 1.1162, 1.3952, 0.8286, 0.8471]

configurations = {
    'possible_mode': ['CNN'],
    # 'possible_DataDim_CNN': [[256, 256], [512, 512], [1024, 1024], [2048, 2048]],
    'possible_DataDim_CNN': [[258, 582]],
    # 'possible_mode': ['NN'],
    # 'possible_DataDim_NN': [[2031, 4613]],
    'possible_kernel_size': [16],
    'possible_stride': [8],
    'possible_n_kernels': [16],
    # 'possible_lr': [0.01, 0.001, 0.0001],
    'possible_lr': [0.001],
    'possible_patience': [1000],
    'possible_n_epochs': [5],
    'possible_batch_size': [16],
    # 'possible_conv_groups': [1, 2],
    # 'possible_activation': ['none', 'square', 'trainable'],
    'possible_activation': ['square'],
    'possible_weighted_loss': ['fixed'], # none, fixed, trainable
    'possible_scheduler': ['cosine'],
    'possible_weight_decay': [1e-4],
    'possible_dropout_p': [0.5],
    'possible_n_folds': [5],
    'possible_model_class': ['cnn']
}


def clean_configurations(configs: [dict]):
    for config in configs:
        # if the stride > the kernel_size, then stride = kernel_size
        if config['stride'] > config['kernel_size']:
            config['stride'] = config['kernel_size']
        
    # Remove the configurations that are duplicated.
    cleaned_configs = [eval(s) for s in {str(d) for d in configs}]

    log.info(f'Cleaned configurations: {len(cleaned_configs)}/{len(configs)}')

    return cleaned_configs


class GenomicCNN(L.LightningModule):
    def __init__(self, config):
        self.config = config
        super().__init__()
        for key, value in config.items():  # Each key-value becomes an attribute
            setattr(self, key, value)

        self.model_class = 'cnn'
        self.scheduler = self.scheduler

        if self.activation == 'trainable':
            self.a = nn.Parameter(torch.randn(1))
            self.b = nn.Parameter(torch.randn(1))

        if self.mode == 'NN':
            self.conv_CNV_output_size = self.n_kernels * ((((config['DataDim_NN'][0]) - self.kernel_size) // self.stride) + 1)
            self.conv_CNV = nn.Conv1d(in_channels=1,
                                    out_channels=self.n_kernels, stride=self.stride,
                                    kernel_size=self.kernel_size, bias=True)

            self.conv_SNV_output_size = self.n_kernels * ((((config['DataDim_NN'][1]) - self.kernel_size) // self.stride) + 1)
            self.conv_SNV = nn.Conv1d(in_channels=1,
                                      out_channels=self.n_kernels, stride=self.stride,
                                      kernel_size=self.kernel_size, bias=True)
        else:
            self.conv_CNV_output_size = self.n_kernels * ((((config['DataDim_CNN'][0]) - self.kernel_size) // self.stride) + 1)
            self.conv_CNV = nn.Conv1d(in_channels=1,
                                      out_channels=self.n_kernels, stride=self.stride,
                                      kernel_size=self.kernel_size, bias=True)

            self.conv_SNV_output_size = self.n_kernels * ((((config['DataDim_CNN'][1]) - self.kernel_size) // self.stride) + 1)
            self.conv_SNV = nn.Conv1d(in_channels=1,
                                      out_channels=self.n_kernels, stride=self.stride,
                                      kernel_size=self.kernel_size, bias=True)
        
        self.spatial_dropout = nn.Dropout2d(p=self.dropout_p)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.conv_CNV_output_size + self.conv_SNV_output_size, len(CLASS_NAMES))

        self.accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=len(CLASS_NAMES))
        self.confusion_matrix = torchmetrics.classification.ConfusionMatrix(task='multiclass', num_classes=len(CLASS_NAMES))
        self.additional_metrics = {
            'micro_auc': torchmetrics.classification.MultilabelAUROC(average='micro', num_labels=len(CLASS_NAMES)),
            'macro_auc': torchmetrics.classification.MultilabelAUROC(average='macro', num_labels=len(CLASS_NAMES)),
            'weighted_auc': torchmetrics.classification.MultilabelAUROC(average='weighted', num_labels=len(CLASS_NAMES))
        }

        self.save_hyperparameters()

        
    def forward(self, x):
        if self.config['mode'] == 'NN':
            x_CNV = self.conv_CNV(x[:, np.newaxis, :self.config['DataDim_NN'][0]])
            x_SNV = self.conv_SNV(x[:, np.newaxis, -self.config['DataDim_NN'][1]:])

            if self.activation == 'square':
                x_CNV = x_CNV ** 2
                x_SNV = x_SNV ** 2
            elif self.activation == 'trainable':
                x_CNV = self.a * x_CNV ** 2 + self.b * x_CNV
                x_SNV = self.a * x_SNV ** 2 + self.b * x_SNV

            x_CNV = x_CNV.unsqueeze(3)  # Now shape is (batch_size, channels, seq_length, 1)
            x_CNV = self.spatial_dropout(x_CNV)
            x_CNV = x_CNV.squeeze(3)  # Revert to original shape

            x_SNV = x_SNV.unsqueeze(3)  # Now shape is (batch_size, channels, seq_length, 1)
            x_SNV = self.spatial_dropout(x_SNV)
            x_SNV = x_SNV.squeeze(3)  # Revert to original shape

        else:
            x_CNV = self.conv_CNV(x[:, np.newaxis, :self.config['DataDim_CNN'][0]])
            x_SNV = self.conv_SNV(x[:, np.newaxis, -self.config['DataDim_CNN'][1]:])

            if self.activation == 'square':
                x_CNV = x_CNV ** 2
                x_SNV = x_SNV ** 2
            elif self.activation == 'trainable':
                x_CNV = self.a * x_CNV ** 2 + self.b * x_CNV
                x_SNV = self.a * x_SNV ** 2 + self.b * x_SNV

            x_CNV = x_CNV.unsqueeze(3)  # Now shape is (batch_size, channels, seq_length, 1)
            x_CNV = self.spatial_dropout(x_CNV)
            x_CNV = x_CNV.squeeze(3)  # Revert to original shape

            x_SNV = x_SNV.unsqueeze(3)  # Now shape is (batch_size, channels, seq_length, 1)
            x_SNV = self.spatial_dropout(x_SNV)
            x_SNV = x_SNV.squeeze(3)  # Revert to original shape

        x = torch.cat((x_CNV, x_SNV), axis = 2)
        x = self.flatten(x)
        x = self.fc1(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        expected = torch.argmax(y, dim=1)
        
        loss = F.cross_entropy(logits, expected)
        acc = self.accuracy(preds, expected)

        for metric in self.additional_metrics.values():
            metric.update(logits, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def on_training_epoch_end(self):
        for metric_name, metric_function in self.additional_metrics.items():
            self.log(f'train_{metric_name}', metric_function.compute(), on_epoch=True, prog_bar=True)
            metric_function.reset()


    def validation_step(self, batch, batch_idx):
        x, y = batch  # y -> [batch_size, n_classes]
        logits = self(x)  # [batch_size, n_classes]
        preds = torch.argmax(logits, dim=1)  # [batch_size]
        expected = torch.argmax(y, dim=1)  # [batch_size]
        
        loss = F.cross_entropy(logits, expected)
        acc = self.accuracy(preds, expected)

        for metric in self.additional_metrics.values():
            metric.update(logits, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):
        for metric_name, metric_function in self.additional_metrics.items():
            self.log(f'val_{metric_name}', metric_function.compute(), on_epoch=True, prog_bar=True)
            metric_function.reset()
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        expected = torch.argmax(y, dim=1)
        
        loss = F.cross_entropy(logits, expected)
        acc = self.accuracy(preds, expected)
        self.confusion_matrix.update(preds, expected)

        for metric in self.additional_metrics.values():
            metric.update(logits, y)

        self.log(f'test_loss_{self.model_type}', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f'test_acc_{self.model_type}', acc, on_step=False, on_epoch=True, prog_bar=True)


    def on_test_epoch_end(self):
        # Compute the confusion matrix
        confmat = self.confusion_matrix.compute()
        # Reset for the next epoch
        self.confusion_matrix.reset()
        
        confmat = confmat.cpu()

        # Normalize the confusion matrix to get percentages
        confmat_percent = confmat / confmat.sum(axis=1, keepdims=True)
        confmat_percent = np.nan_to_num(confmat_percent)  # Convert NaNs to 0

        # Create a figure with two subplots (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

        # Plot the count-based confusion matrix
        sns.heatmap(confmat, annot=True, fmt='g', ax=ax1, cmap=plt.cm.Blues, 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax1.set_xlabel('Predicted labels')
        ax1.set_ylabel('True labels')
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.xaxis.set_tick_params(rotation=45)
        ax1.yaxis.set_tick_params(rotation=45)

        # Plot the percentage-based confusion matrix
        sns.heatmap(confmat_percent, annot=True, fmt=".2%", ax=ax2, cmap=plt.cm.Blues, 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax2.set_xlabel('Predicted labels')
        ax2.set_ylabel('True labels')
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.xaxis.set_tick_params(rotation=45)
        ax2.yaxis.set_tick_params(rotation=45)

        # Tight layout to ensure no overlap
        plt.tight_layout()
        plt.close(fig)

        for metric_name, metric_function in self.additional_metrics.items():
            self.log(f'test_{metric_name}_{self.model_type}', metric_function.compute(), on_epoch=True, prog_bar=True)
            metric_function.reset()


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler == 'cosine':
            scheduler = {
                'scheduler': CosineAnnealingLR(optimizer, T_max=int(self.n_epochs), eta_min=0),
                'name': 'learning_rate',
                'interval': 'epoch',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        else:
            return optimizer
    

def compute(config, DEVICE, N_EXPERIMENTS, NAME_PREFIX=''):
    # Put these here, so that even in multiprocessing they are applied.
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*available.*")

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)  # To make torch use only one CPU process per trial

    config['device_name'] = DEVICE

    log.info('Starting case.', **config)

    if config['mode'] == 'NN':
        N_CNV = config['DataDim_NN'][0]
        N_SNV = config['DataDim_NN'][1]
    elif config['mode'] == 'CNN':
        N_CNV = config['DataDim_CNN'][0]
        N_SNV = config['DataDim_CNN'][1]


    for n_experiment in range(N_EXPERIMENTS):
        folds = create_stratified_folds(N_CNV, N_SNV, config['mode'], batch_size=config['batch_size'], num_folds=config['n_folds'])
        for i, loaders in enumerate(folds):
            train_loader, val_loader, test_loader = loaders

            # Create the model
            config['fold_index'] = i
            cnn = GenomicCNN(config)

            # Train.
            early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=config['patience'], verbose=False, mode='min')
            checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")
            lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

            model_name = "_".join(f"{key}{value}" for key, value in config.items())

            trainer = L.Trainer(accelerator=DEVICE,
                                callbacks=[early_stop_callback, checkpoint_callback, lr_monitor_callback], 
                                max_epochs=config['n_epochs'], 
                                enable_progress_bar=True, enable_model_summary=True)
            
            
            trainer.fit(model=cnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
            cnn = GenomicCNN.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
            cnn.model_type = 'best_epoch'
            trainer.test(model=cnn, dataloaders=test_loader)

            torch.save(cnn, f"cnn_{N_CNV}_{N_SNV}.pt")

            batch = next(iter(test_loader))
            with open(f'sample_data_single.pkl', 'wb') as f:
                pickle.dump(batch, f)
            
            break


    log.info('Finished case.', **config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch the experiments.')
    parser.add_argument('--device', type=str, required=True, choices=['cpu', 'gpu'], help='Device: either `cpu` or `gpu`.')
    parser.add_argument('--n_jobs', type=int, required=False, help='Number of parallel jobs: 1 deactivates multiprocessing.')
    parser.add_argument('--n_experiments', type=int, required=False, default=1, help='Number of experiments to run for each case.')

    args = parser.parse_args()

    keys = [key.replace('possible_', '') for key in configurations]
    values = list(configurations.values())

    # Computing the cartesian product of the values
    dynamic_combinations = product(*values)

    # Creating a list of dictionaries with these combinations, associating them with the original keys
    combinations = [{keys[i]: value for i, value in enumerate(comb)} for comb in dynamic_combinations]

    # Clean combinations
    combinations = clean_configurations(combinations)

    Parallel(n_jobs=args.n_jobs)(delayed(compute)(c, args.device, 1) for c in combinations)
