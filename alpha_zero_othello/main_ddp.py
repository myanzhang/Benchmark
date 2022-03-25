
# dor ddp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import logging

import coloredlogs

from Coach_ddp import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.pytorch.NNet_ddp import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# args = dotdict({
#     'numIters': 1000,
#     'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
#     'tempThreshold': 15,        #
#     'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
#     'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
#     'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
#     'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
#     'cpuct': 1,

#     'checkpoint': './temp/',
#     'load_model': False,
#     'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
#     'numItersForTrainExamplesHistory': 20,

#     # for ddp
#     'local_rank': 0,
# })


import argparse
parser = argparse.ArgumentParser(description='Alpha zero.')
parser.add_argument('--numIters', type=int, default=1000, help='number iters for training')
parser.add_argument('--numEps', type=int, default=50, help='Number of complete self-play games to simulate during a new iteration.')
parser.add_argument('--tempThreshold', type=int, default=15, help='todo.')
parser.add_argument('--updateThreshold', type=float, default=0.6, help='During arena playoff, new neural net will be accepted if threshold or more of games are won.')
parser.add_argument('--maxlenOfQueue', type=int, default=2000000, help='Number of game examples to train the neural networks.')
parser.add_argument('--numMCTSSims', type=int, default=25, help='Number of games moves for MCTS to simulate.')
parser.add_argument('--arenaCompare', type=int, default=40, help='Number of games to play during arena play to determine if new net will be accepted..')
parser.add_argument('--cpuct', type=int, default=1, help='...')
parser.add_argument('--checkpoint', type=str, default='./temp/', help='...')
parser.add_argument('--load_model', type=bool, default=False, help='...')
parser.add_argument('--load_folder_file', type=str, default=('/dev/models/8x100x50','best.pth.tar'), help='...')
parser.add_argument('--numItersForTrainExamplesHistory', type=int, default=100, help='...')

parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training.')
args = parser.parse_args()


def main():
    
    # for ddp
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    log.info('Loading %s...', Game.__name__)
    g = Game(8)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
