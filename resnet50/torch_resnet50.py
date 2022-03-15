'''
use imagenet:
python3 torch_resnet50.py --train-dir 'path_to_imagenet'
'''


import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import math
from tqdm import tqdm

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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def log(s, nl=True):
    print(s, end='\n' if nl else '')


def train_synthetic_(model, optimizer, data, target):
    log('Running benchmark...')
    for epoch in range(args.epochs):
        with tqdm(total=args.num_batches_per_epoch,
                  desc='Train Epoch #{}'.format(epoch + 1)) as t:
            for i in range(args.num_batches_per_epoch):
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()

                # Gradient is applied across all ranks
                optimizer.step()
                t.set_postfix({'loss': loss})
                t.update(1)


def train_imagenet(model, optimizer, train_loader):
    for epoch in range(0, args.epochs):
        model.train()
        with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1)) as t:
            for batch_idx, (data, target) in enumerate(train_loader):
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                # Split data into sub-batches of size batch_size
                for i in range(0, len(data), args.batch_size):
                    data_batch = data[i:i + args.batch_size]
                    target_batch = target[i:i + args.batch_size]
                    output = model(data_batch)
                    loss = F.cross_entropy(output, target_batch)
                    # Average gradients among sub-batches
                    loss.div_(math.ceil(float(len(data)) / args.batch_size))
                    loss.backward()
                # Gradient is applied across all ranks
                optimizer.step()
                t.set_postfix({'loss': loss})
                t.update(1)


def main():
    # base information.
    log('Model: %s' % 'ResNet50')
    log('Batch size: %d' % args.batch_size)
    device = 'GPU' if args.cuda else 'CPU'
    log('Number of %ss: %d' % (device, 1))
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(args.seed)
    
    # Set up standard ResNet-50 model. & Move model to GPU.
    model = models.resnet50()
    if args.cuda:
        model.cuda()
    
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)
    
    if args.train_dir is not None:
        log('Use imagenet dataset.')
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
        train_dataset = datasets.ImageFolder(args.train_dir,
                transform=transforms.Compose([transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=None, **kwargs)
        train_imagenet(model, optimizer, train_loader)
    else:
        log('Use synthetic dataset.')
        data = torch.randn(args.batch_size, 3, 224, 224)
        target = torch.LongTensor(args.batch_size).random_() % 1000
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        train_synthetic_(model, optimizer, data, target)


if __name__ == '__main__':
    main()
