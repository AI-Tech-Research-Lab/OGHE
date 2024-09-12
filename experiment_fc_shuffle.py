import argparse
from collections import defaultdict
import os
import pickle
import re
import uuid
import lightning as L
import logging
import numpy as np
import pandas as pd
import structlog
from itertools import product
import warnings
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from joblib import Parallel, delayed


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

from data_utils import create_stratified_folds


log = structlog.get_logger()

CLASS_NAMES = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']
CLASS_WEIGHTS = [0.8179, 1.0483, 0.3300, 1.4160, 0.8214, 0.9632, 1.4160, 1.1162, 1.3952, 0.8286, 0.8471]
WANDB_PROJECT_NAME = 'FC_KFold_Ale_2024_04_06'

configurations = {
    # 'possible_mode': ['CNN'],
    # 'possible_DataDim_CNN': [[256, 256], [512, 512], [1024, 1024], [2048, 2048]],
    'possible_mode': ['NN'],
    'possible_DataDim_NN': [[258, 582], [514, 1152], [710, 1197], [1024, 2286], [2031, 4613]],
    'possible_lr': [0.001, 0.0001],
    'possible_patience': [1000],
    'possible_n_epochs': [200],
    'possible_batch_size': [4, 8, 16, 32],
    'possible_weighted_loss': [True],
    'possible_scheduler': ['cosine', 'none'],
    'possible_weight_decay': [0, 1e-4, 5e-4],
    'possible_n_folds': [5],
    'possible_model_class': ['fc']
}


class GenomicCNN(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.model_class = 'fc'
        self.lr = config['lr']
        self.n_epochs = config['n_epochs']
        self.weight_decay = config['weight_decay']

        if config['weighted_loss']:
            self.loss = lambda logits, expected: F.cross_entropy(logits, expected, weight=torch.tensor(CLASS_WEIGHTS))
        else:
            self.loss = lambda logits, expected: F.cross_entropy(logits, expected)
        self.scheduler = config['scheduler']

        if config['mode'] == 'NN':
            N_CNV = config['DataDim_NN'][0]
            N_SNV = config['DataDim_NN'][1]
        else:
            N_CNV = config['DataDim_CNN'][0]
            N_SNV = config['DataDim_CNN'][1]

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(N_CNV+N_SNV, len(CLASS_NAMES))

        self.accuracy = torchmetrics.classification.Accuracy(task='multiclass', num_classes=len(CLASS_NAMES))
        self.confusion_matrix = torchmetrics.classification.ConfusionMatrix(task='multiclass', num_classes=len(CLASS_NAMES))
        self.additional_metrics = {
            'micro_auc': torchmetrics.classification.MultilabelAUROC(average='micro', num_labels=len(CLASS_NAMES)),
            'macro_auc': torchmetrics.classification.MultilabelAUROC(average='macro', num_labels=len(CLASS_NAMES)),
            'weighted_auc': torchmetrics.classification.MultilabelAUROC(average='weighted', num_labels=len(CLASS_NAMES))
        }

        self.save_hyperparameters()

    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)

        if not self.training:
            x = F.softmax(x, dim=1)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        expected = torch.argmax(y, dim=1)
        
        loss = self.loss(logits, expected)
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
        
        loss = self.loss(logits, expected)
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
        
        loss = self.loss(logits, expected)
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

        # Log the combined confusion matrix image to Wandb
        self.logger.experiment.log({f"test_confusion_matrix_{self.model_type}": wandb.Image(fig)})
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

    for experiment in range(N_EXPERIMENTS):
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
            wandb_path = f"wandb_stuff/{uuid.uuid4()}"
            os.mkdir(wandb_path)

            explogger = WandbLogger(save_dir=wandb_path, project=WANDB_PROJECT_NAME, name=f"{NAME_PREFIX}_{model_name}", entity='paper_ecai')
            
            trainer = L.Trainer(accelerator=DEVICE, logger=explogger, 
                                callbacks=[early_stop_callback, checkpoint_callback, lr_monitor_callback], 
                                max_epochs=config['n_epochs'], 
                                enable_progress_bar=True, enable_model_summary=True)
            
            
            trainer.fit(model=cnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
            cnn.model_type = 'last_epoch'
            trainer.test(model=cnn, dataloaders=test_loader)

            cnn = GenomicCNN.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
            cnn.model_type = 'best_epoch'
            trainer.test(model=cnn, dataloaders=test_loader)

            wandb.finish()
            os.system(f'rm -rf {wandb_path}')

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

    
    Parallel(n_jobs=args.n_jobs)(delayed(compute)(c, args.device, 1) for c in combinations)

    ## Now that the hyperparameters search has been done, we can fetch the "best" hyperparameters set (based on val loss),
    ## and compute the final K-Fold experiment for 10 times (for mean and variance analysis).

    api = wandb.Api()
    combinations = []

    possible_data_dims = configurations['possible_DataDim_NN'] if 'NN' in configurations['possible_mode'] else configurations['possible_DataDim_CNN']
    

    wandb_runs = api.runs(f"paper_ecai/{WANDB_PROJECT_NAME}")

    def fetch_run_data(run_id):
        api = wandb.Api()
        run = api.run(f"paper_ecai/{WANDB_PROJECT_NAME}/{run_id}")
        return {
                'name': run.display_name,
                'val_loss': run.history()['val_loss'].min(),
                'config': run.config['config']
            }

    runs = Parallel(n_jobs=10)(delayed(fetch_run_data)(run.id) for run in wandb_runs)

    # with open('temp', 'wb') as f:
    #     pickle.dump(runs, f)
        
    # with open('temp', 'rb') as f:
    #     runs = pickle.load(f)


    # For each data_dim...
    for data_dim in possible_data_dims:
        # Compile a regex pattern for this data_dim
        pattern = re.compile(f".*\{str(data_dim)[:-1]}\].*")
        
        # Filter runs for this specific data_dim
        this_data_dim_runs = [run for run in runs if pattern.match(run['name'])]
        
        # Now this_data_dim_runs is a list of dicts with all the runs of this data_dim.
        this_data_dim_runs = pd.DataFrame(this_data_dim_runs)

        # Now group the runs by name, and compute the mean of the val loss.
        grouped_data = defaultdict(list)
        for _, row in this_data_dim_runs.iterrows():
            key = row['name']
            grouped_data[key].append(row)
        
        results = []

        for name, rows in grouped_data.items():
            mean_val_loss = np.mean([row['val_loss'] for row in rows])

            results.append({
                'name': name,
                'mean_val_loss': mean_val_loss,
                'config': rows[0]['config']
            })
        
        results_df = pd.DataFrame(results)
        temp_results = results_df.sort_values('mean_val_loss')
        best = temp_results.iloc[0, :]

        config = best['config']

        log.info(f"Best run: {best['name']}")
        log.info(f"Best run Validation Loss: {best['mean_val_loss']}")
        log.info(f"Config of the best run: {config}")
        combinations.append(config)


    Parallel(n_jobs=args.n_jobs)(delayed(compute)(c, args.device, args.n_experiments, 'best_') for c in combinations)
