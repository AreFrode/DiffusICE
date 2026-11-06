import torch
import os
import csv
import lightning.pytorch as L

from math import sqrt

from WoofLoader import WoofLoader
from LitDiffusionUnet import LitDiffusionEDM

def config_writer(config_path, config):
    with open(f"{config_path}/{config['model_name']}.csv", "w") as config_out:
        w = csv.DictWriter(config_out, config.keys())
        w.writeheader()
        w.writerow(config)


def get_default_config():
    return {
        # Model settings
        'model_name': None,
        'channels': [32, 64, 128, 256],
        'encoder_block_multiples': 2,
        'decoder_block_multiples': 2,
        'num_bottleneck_blocks': 2,
        'embedding_dim': 128,
        'embedding_hidden_dim': 256,
        'padding_mode': 'reflect',

        # Dataloader settings
        'batch_size': 32,
        'mode': '19-23-37-89',                # Currently 19-37-89 (all data)

        # Optimizer settings
        # Scale LR with the square root of GPU count
        'total_gradient_updates': 100000,
        'num_warmup_updates': 5000,
        'base_learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'accumulating_grad_batches': 1,

        # lr-schdeduler settings
        'min_lr': 1e-6,
        'early_stopping_patience': 30,
        'check_val_every_epoch': 10,

        # Regularization settings
        'max_drop_path_rate': .1,
        'attn_dropout': .1,
        'mlp_dropout': .1,

        # Misc.
        'lowestloss_epoch': 'N/A',
        'lowestvalloss_epoch': 'N/A',
    }


def main():
    torch.set_float32_matmul_precision("high")

    dev_run = True

    config = get_default_config()

    num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE"))
    num_nodes = int(os.environ.get("SLURM_NNODES"))
    total_gpus = int(num_gpus * num_nodes)

    train_dataloader, train_samples = WoofLoader()

    config['learning_rate'] = config['base_learning_rate'] * sqrt(total_gpus)

    config['steps_per_epoch'] = train_samples // (config['batch_size'] * total_gpus * config['accumulating_grad_batches'])
    config['epochs'] = config['total_gradient_updates'] // config['steps_per_epoch']

    config['warmup_epochs'] = config['num_warmup_updates'] // config['steps_per_epoch']

    model = LitDiffusionEDM(
        config = config
    )

    callbacks = [L.callbacks.TQDMProgressBar()]


    trainer = L.Trainer(
        accelerator = "auto",
        fast_dev_run = dev_run,
        max_epochs = config["epochs"],
        precision = "bf16-mixed",
        devices = int(os.environ.get("SLURM_GPUS_ON_NODE")),
        num_nodes = int(os.environ.get("SLURM_NNODES")),
        gradient_clip_val = 1.0,
        strategy = "ddp_find_unused_parameters_true",
        callbacks = callbacks
    )

    config_writer("outputs/configs", config)

    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()
