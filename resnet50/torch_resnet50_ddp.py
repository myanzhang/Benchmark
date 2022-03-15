'''
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="xxxxx" \
    --master_port=23456 torch_resnet50_ddp.py --train-dir 'path-to-imagente.'

AMD:
HIP_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr="x.x.x.x" --master_port=23456 torch_resnet50_ddp.py --train-dir 'path-to-imagente.'

NVIDIA:
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr="x.x.x.x" --master_port=23456 torch_resnet50_ddp.py --train-dir 'path-to-imagente.'
'''


import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import math
from tqdm import tqdm

# DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Training settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet/Synthetic Example', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-dir', default=None, help='path to training data')
parser.add_argument('--batch-size', type=int, default=300, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125, help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005, help='weight decay')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num-batches-per-epoch', type=int, default=10, help='number of batches per benchmark epoch.')

# DDP
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def log(s, nl=True):
    print(s, end='\n' if nl else '')


def train_imagenet(model, optimizer, train_loader, train_sampler):
    def accuracy(output, target):
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        return pred.eq(target.view_as(pred)).cpu().float().mean()
    
    for epoch in range(0, args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1)) as t:
            for _, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                # Split data into sub-batches of size batch_size
                for i in range(0, len(data), args.batch_size):
                    data_batch = data[i:i + args.batch_size]
                    target_batch = target[i:i + args.batch_size]
                    output = model(data_batch)
                    acc = accuracy(output, target_batch)
                    loss = F.cross_entropy(output, target_batch)
                    # Average gradients among sub-batches
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                    loss.backward()
                # Gradient is applied across all ranks
                optimizer.step()
                t.set_postfix({'loss': loss.item(), 'accuracy': 100. * acc.item()})
                t.update(1)


def main():
    dist.init_process_group(backend="nccl", init_method='env://')
    
    # base information.
    log('Model: %s' % 'ResNet50')
    log('Batch size: %d' % args.batch_size)
    device = 'GPU' if args.cuda else 'CPU'
    log('Number of %ss: %d' % (device, dist.get_world_size()))
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.manual_seed(args.seed)
    
    # Set up standard ResNet-50 model. & Move model to GPU.
    model = models.resnet50()
    if args.cuda:
        model.cuda()
    
    # DDP
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
    
    log('Use imagenet dataset.')
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'
    
    train_dataset = datasets.ImageFolder(args.train_dir,
            transform=transforms.Compose([transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    train_imagenet(model, optimizer, train_loader, train_sampler)


if __name__ == '__main__':
    main()
