import os
import time
import torch

from torchvision import models
from torch.utils.data import DataLoader

from dataset import DigiFaceDataset
from pathlib import Path

# XLA imports
import torch_xla.core.xla_model as xm

# Global constants
EPOCHS = 4
WARMUP_STEPS = 2
BATCH_SIZE = 8
THREADS = 4
DATASET_DIR = Path('/home/ubuntu/app/dataset')

# Load MNIST train dataset
train_dataset = DigiFaceDataset(DATASET_DIR.joinpath(Path("train")), image_size = (224, 224), pre_transforms = None)

def main():
    # Prepare data loader
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = THREADS)

    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    device = 'xla'

    # Move model to device and declare optimizer and loss function
    model = models.resnet18(weights = None).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.TripletMarginLoss(margin = 0.2, p = 2)

    # Run the training loop
    print('----------Training ---------------')
    model.train()
    for epoch in range(EPOCHS):
        start = time.time()
        for idx, (a_img, p_img, n_img) in enumerate(train_loader):
            optimizer.zero_grad()
            
            a_img, p_img, n_img = a_img.to(device), p_img.to(device), n_img.to(device)
            
            a_emb = model(a_img)
            p_emb = model(p_img)
            n_emb = model(n_img)
            loss = loss_fn(a_emb, p_emb, n_emb)
            
            loss.backward()
            optimizer.step()
            xm.mark_step() # XLA: collect ops and run them in XLA runtime
            if idx < WARMUP_STEPS: # skip warmup iterations
                start = time.time()
    # Compute statistics for the last epoch
    interval = idx - WARMUP_STEPS # skip warmup iterations
    throughput = interval / (time.time() - start)
    print("Train throughput (iter/sec): {}".format(throughput))
    print("Final loss is {:0.4f}".format(loss.detach().to('cpu')))

    # Save checkpoint for evaluation
    os.makedirs("ckpts", exist_ok=True)
    checkpoint = {'state_dict': model.state_dict()}
    # XLA: use xm.save instead of torch.save to ensure states are moved back to cpu
    # This can prevent "XRT memory handle not found" at end of test.py execution
    xm.save(checkpoint,'ckpts/checkpoint.pt')

    print('----------End Training ---------------')

if __name__ == '__main__':
    main()

