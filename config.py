import os
import argparse
import torch

parser = argparse.ArgumentParser(description="arguments for project")

# arguments about training
use_gpu = True
gpu_count = torch.cuda.device_count()
device = f"cuda:{','.join(map(str,list(range(gpu_count))))}" if use_gpu else 'cpu'
parser.add_argument('--device', default=device, help="device to train model")
parser.add_argument('--batch_size', type=int, default=1, help='batch size while training')
parser.add_argument('--epoch', type=int, default=3000, help="epochs to train")
parser.add_argument('--num_workers', type=int, default=3, help='Num of workers to load data')
parser.add_argument('--on_gpu', type=bool, default=use_gpu, help='Run on GPU or not')
parser.add_argument('--slice_num', type=int, default=40, help='Selected num of slices in a CT')
parser.add_argument('--restore_training', type=bool, default=False, help="Restore training or not")


# arguments about path info
proj_root = "/home/gzy/medical/abdominal-multi-organ-segmentation/"
# model_dir = os.path.join(proj_root, "module")
dataset_dir = os.path.join(proj_root, "dataset")
train_dataset_dir = os.path.join(dataset_dir, "train")
val_dataset_dir = os.path.join(dataset_dir, "val")
test_dataset_dir = os.path.join(dataset_dir, "test")
output_dir = os.path.join(proj_root, "output")
model_dir = os.path.join(output_dir, "output/module")
model_name = "net990-2.000-1.198.pth"

parser.add_argument("--dataset_dir", type=str, default=dataset_dir, help="Dataset dir path")
parser.add_argument("--model_dir", type=str, default=model_dir, help="Directory where models will be dumped/loaded")
parser.add_argument("--train_dataset_dir", type=str, default=train_dataset_dir, help="Train dataset dir")
parser.add_argument("--val_dataset_dir", type=str, default=val_dataset_dir, help="Val dataset dir")
parser.add_argument("--test_dataset_dir", type=str, default=test_dataset_dir, help="Test Dataset dir")
parser.add_argument("--output_dir", type=str, default=output_dir, help="Output dir")
parser.add_argument('--checkpoint_path', type=str, default=os.path.join(model_dir, model_name), help="Checkpoint file path")

config = parser.parse_args()
