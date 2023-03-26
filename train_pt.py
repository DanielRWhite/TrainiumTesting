import torch

#from torchvision import transformers
from dataset import DigiFaceDataset
from pathlib import Path

#import torchvision
import timm

"""
XLA Stuff
"""
import torch_xla
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_xla.distributed.xla_backend
import torch_xla.test.test_utils as test_utils

"""
        args = argparse.ArgumentParser()
        args.add_argument('--max_epochs', type = int, default = 200)
        args.add_argument('--batch_size', type = int, default = 8)
        args.add_argument('--threads', type = int, default = 4)
        args.add_argument('--dataset', type = Path)
        hparams = args.parse_args()
"""

MAX_EPOCHS = 256
BATCH_SIZE = 8
THREADS = 4
DATASET_DIR = Path('/home/ubuntu/app/dataset')

def _train_update(device, step, loss, tracker, epoch, writer):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch,
      summary_writer=writer)

def main():
        torch.manual_seed(1205)
        
        device = xm.xla_device()
        
        model = timm.create_model('maxvit_rmlp_pico_rw_256.sw_in1k', pretrained = False, num_classes = 0)
        model = model.to(device)
        
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training = False)
        
        train_dataset = DigiFaceDataset(DATASET_DIR.joinpath(Path("train")), pre_transforms = transforms)
        train_sampler = None
        
        if xm.xrt_world_size() > 0:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_dataset,
                        num_replicas = xm.xrt_world_size(),
                        rank = xm.get_ordinal(),
                        shuffle = True
                )
                
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, sampler = train_sampler, shuffle = False if train_sampler is not None else True, num_workers = THREADS)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
        loss_fn = torch.nn.TripletMarginLoss(margin = 0.2, p = 2)
        
        writer = None
        if xm.is_master_ordinal():
                writer = test_utils.get_summary_writer("/home/ubuntu/app/logs")
        
        if xm.is_master_ordinal():
                print('---------- Training ---------------')

        def train_loop(loader, epoch):
                tracker = xm.RateTracker()
                model.train()
                print('LENGTH LOADER: ', len(loader))
                for step, (anchor_img, pos_img, neg_img) in enumerate(loader):
                        optimizer.zero_grad()
                        
                        a_emb = model(anchor_img)
                        p_emb = model(pos_img)
                        n_emb = model(neg_img)
                        
                        loss = loss_fn(a_emb, p_emb, n_emb)
                        loss.backward()
                        xm.optimizer_step(optimizer)
                        tracker.add(BATCH_SIZE)
                        if step % 8 == 0:
                                xm.add_step_closure(
                                        _train_update,
                                        args=(device, step, loss, tracker, epoch, writer),
                                        run_async=True
                                )
        
        train_device_loader = pl.MpDeviceLoader(train_loader, device)
        for epoch in range(1, MAX_EPOCHS + 1):
                xm.master_print(f'Epoch {epoch:00}/{MAX_EPOCHS:00} begin')
                train_loop(train_device_loader, epoch)
                xm.master_print(f'Epoch {epoch:00}/{MAX_EPOCHS:00} finish')
                
        test_utils.close_summary_writer(writer)
        
        return

def _mp_fn(index):
        torch.set_default_tensor_type(torch.FullTensor)
        main()
        
if __name__ == "__main__":
        xmp.spawn(_mp_fn, nprocs=2)