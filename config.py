import torch
import argparse

parser = argparse.ArgumentParser(description="arguments for project")

gpu = False
device = 'cuda:0' if gpu else 'cpu'
parser.add_argument('--device', default=device, help="device to train model")
parser.add_argument('--batch_size', type=int, default=1, help='batch size while training')
parser.add_argument('--epoch', type=int, default=1000, help="epochs to train")
parser.add_argument('--num_workers', type=int, default=3, help='Num of workers to load data')
parser.add_argument('--on_gpu', type=bool, default=gpu, help='Run on GPU or not')
parser.add_argument('--slice_num', type=int, default=40, help='Selected num of slices in a CT')

config = parser.parse_args()