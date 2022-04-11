import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
from .OthelloNNet import OthelloNNet as onnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 20,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


class ExamplesDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, i):
        boards, pis, vs = self.examples[i]
        boards = torch.from_numpy(np.array(boards).astype(np.float32))
        target_pis = torch.from_numpy(np.array(pis).astype(np.float32))
        target_vs = torch.from_numpy(np.array(vs).astype(np.float32))
        return boards, target_pis, target_vs

    def __len__(self):
        return len(self.examples)


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = torch.jit.script(onnet(game, args))
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet = self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()

            ds = ExamplesDataset(examples)
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
            t = tqdm(dl, desc='Training Net')
            
            with torch.autograd.profiler.profile(False, use_cuda=True, with_stack=True) as p:
                with torch.jit.fuser('fuser2'):
                    for boards, target_pis, target_vs in t:
                        if args.cuda:
                            boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                        # compute output
                        out_pi, out_v = self.nnet(boards)
                        l_pi = loss_pi(target_pis, out_pi)
                        l_v = loss_v(target_vs, out_v)
                        total_loss = l_pi + l_v

                        # compute gradient and do SGD step
                        optimizer.zero_grad(True)
                        total_loss.backward()
                        optimizer.step()
            if p:
                p.export_chrome_trace("autograd_trace.json")
                print(p.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
