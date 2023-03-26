import torch
torch.set_float32_matmul_precision('medium')

from dataset import DigiFaceDataset
from torch import optim, nn, utils
from pathlib import Path

import torchvision
import pytorch_lightning as pl
import argparse
#import timm

class LightningPipeline(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, _):
        # training_step defines the train loop.
        # it is independent of forward
        inputs, positives, negatives = batch
        
        outputs = self.model(inputs)
        e_pos = self.model(positives)
        e_neg = self.model(negatives)
        
        loss = nn.functional.triplet_margin_loss(outputs, e_pos, e_neg, margin=0.8, p=2)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)        
        return loss

    def validation_step(self, batch, _):
        # validation_step defines the val loop.
        # it is independent of forward
        inputs, positives, negatives = batch
        
        outputs = self.model(inputs)
        e_pos = self.model(positives)
        e_neg = self.model(negatives)
        
        loss = nn.functional.triplet_margin_loss(outputs, e_pos, e_neg, margin=0.8, p=2)

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer
        
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--accelerator', type = str, choices = [ 'auto', 'cpu', 'gpu', 'tpu', 'ipu', 'hpu', 'mps' ], default = 'auto')
    args.add_argument('--devices', default = 'auto')
    args.add_argument('--num_nodes', type = int, default = 1)
    args.add_argument('--max_epochs', type = int, default = 8)
    args.add_argument('--batch_size', type = int, default = 32)
    args.add_argument('--num_workers', type = int, default = 8)
    args.add_argument('--dataset', type = Path)
    hparams = args.parse_args()
    
    #model = timm.create_model('maxvit_rmlp_pico_rw_256.sw_in1k', pretrained = False, num_classes = 0)
    model = torchvision.models.efficientnet_v2_s(weights = None)
    
    pipeline = LightningPipeline(model)
    
    #data_config = timm.data.resolve_model_data_config(model)
    #transforms = timm.data.create_transform(**data_config, is_training = False)
    
    pre_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((384, 384)),
        torchvision.transforms.CenterCrop((384, 384)),
        torchvision.transforms.ToTensor()
    ])
    
    train_dataset = DigiFaceDataset(hparams.dataset.joinpath("train"), pre_transforms = pre_transforms)
    val_dataset = DigiFaceDataset(hparams.dataset.joinpath("val"), pre_transforms = pre_transforms)

    train_loader, val_loader = (
        utils.data.DataLoader(train_dataset, batch_size = hparams.batch_size, num_workers = hparams.num_workers),
        utils.data.DataLoader(val_dataset, batch_size = hparams.batch_size, num_workers = hparams.num_workers),
    )
    
    cb_checkpoint = (
        pl.callbacks.ModelCheckpoint(dirpath = './ckpt', filename='{epoch}-{val_loss:.2f}')
    )
    
    trainer = pl.Trainer(
        accelerator = hparams.accelerator,
        devices = hparams.devices,
        num_nodes = hparams.num_nodes,
        max_epochs = hparams.max_epochs,
        callbacks = [ cb_checkpoint ],
        precision = 'bf16'
    )
    
    trainer.fit(
        model = pipeline,
        train_dataloaders = train_loader,
        val_dataloaders = val_loader,
    )