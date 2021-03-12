import torch
import argparse

parser = argparse.ArgumentParser(description="arguments for project")

parser.add_argument('--device', default='cpu', help="device to train model")
parser.add_argument('--batch_size', type=int, default=1, help='batch size while training')
parser.add_argument('--epoch', type=int, default=1000, help="epochs to train")

config = parser.parse_args()