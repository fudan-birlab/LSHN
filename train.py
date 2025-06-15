import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np

from torchvision import datasets, transforms

from model import LSHN

def generate_mask(batch):
    mask = torch.zeros_like(batch, device=batch.device)
    mask[:,:mask.shape[-1]//2] = 1.
    return mask

parser = argparse.ArgumentParser(description="Script parameters")

parser.add_argument('--dataset_name', type=str, default='CIFAR10',
                    choices=['MNIST', 'CIFAR10'],
                    help="Name of the dataset: MNIST or CIFAR10")
parser.add_argument('--emb_size', type=int, default=128,
                    help="Embedding size")
parser.add_argument('--num_pattern', type=int, default=100,
                    help="Number of patterns")
parser.add_argument('--total_step', type=int, default=20000,
                    help="Total training steps")
parser.add_argument('--random_idx', type=int, default=0,
                    help="Random index")

args = parser.parse_args()

# 使用示例
print("Dataset:", args.dataset_name)
print("Embedding Size:", args.emb_size)
print("Num Pattern:", args.num_pattern)
print("Total Step:", args.total_step)
print("Random Index:", args.random_idx)

dataset_name = args.dataset_name
emb_size = args.emb_size
num_pattern = args.num_pattern
total_step = args.total_step
random_idx = args.random_idx

ckpt_path = f'./outputs/attractor-{dataset_name}-ckpt/N={emb_size}-npat={num_pattern}-{total_step}step-{random_idx}.pt'
os.makedirs(f'./outputs/attractor-{dataset_name}-ckpt/', exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)) if dataset_name == 'MNIST' else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if dataset_name == 'MNIST':
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
elif dataset_name == 'CIFAR10':
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

whole_patterns = []
labels = []

for i in range(num_pattern):
    item = dataset.__getitem__(i)
    whole_patterns.append(item[0].permute(1,2,0).flatten()[None])
    labels.append(item[1])

whole_patterns = torch.cat(whole_patterns, 0).flatten(1, -1).cuda()
labels = np.array(labels)

attribute_list = ['latent']
model = LSHN(attribute_list, emb_size, image_size=whole_patterns.shape[-1]).cuda()
model.constrain_attractor()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=1e-2)
loss_scale = 1.
noise_scale = 0.5
target_loss_scale = 1.

if not os.path.exists(ckpt_path):
    model.train()
    model.set_dt(1.)

    for i in range(total_step):
        if (i+1) % 500 == 0:
            loss_scale *= 0.9
            target_loss_scale *= 2.
            if target_loss_scale > 20.:
                target_loss_scale = 20.

        batch = whole_patterns.clone()

        mask = generate_mask(batch)
        masked_batch = batch * mask

        # # only half-mask
        # noisy_batch = masked_batch

        # half-mask & gaussian noise
        perturb = torch.normal(torch.zeros_like(batch), torch.rand_like(batch))
        gaussian_batch = torch.clamp(torch.abs((batch + 1) / 2. + perturb), 0, 1) * 2. - 1
        batch = torch.cat([batch, batch], 0)
        noisy_batch = torch.cat([masked_batch, gaussian_batch], 0)

        target_attractors = model.image_to_attractor(batch)
        noisy_attractors = model.image_to_attractor(noisy_batch)
        recon_image = model.attractor_to_image(target_attractors)
        recon_image_noisy = model.attractor_to_image(noisy_attractors)

        target_loss = - target_attractors.abs().mean() - target_attractors.abs().min(-1).values.mean()
        attractor_loss = model.attractor_loss(noisy_attractors, target_attractors)

        prediction_loss = F.l1_loss(batch, recon_image) + F.mse_loss(batch, recon_image)
        prediction_loss_noisy = F.l1_loss(noisy_batch, recon_image_noisy) + F.mse_loss(noisy_batch, recon_image_noisy)

        loss = loss_scale * (target_loss * target_loss_scale + attractor_loss + 5. * (prediction_loss + prediction_loss_noisy))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.constrain_attractor()

        if (i+1) % 100 == 0:
            with torch.no_grad():
                print(f'step {i} - target_loss : {target_loss.item()}; attractor_loss : {attractor_loss.item()}; prediction_loss : {prediction_loss.item()}')

    model.save_model(ckpt_path)
else:
    model.load_state_dict(torch.load(ckpt_path))
