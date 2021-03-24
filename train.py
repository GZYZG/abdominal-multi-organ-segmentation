"""
训练脚本
"""

import os
import time

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from net.ResUnet_dice import net
from loss.ava_Dice_loss import DiceLoss
from data_prepare.dataset import train_ds
from utils.utils import *
import sys
from config import config

if __name__ == "__main__":
    # 定义超参数
    on_server = True
    device = config.device
    print(f"{sys.argv[0]}\n\n")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # if on_server is False else '1,2,3'
    cudnn.benchmark = True
    Epoch = config.epoch
    leaing_rate = 1e-4

    batch_size = config.batch_size
    num_workers = config.num_workers  # 1 if on_server is False else 1
    pin_memory = False if on_server is False else True

    net = net.to(device)
    if config.on_gpu:
        net = torch.nn.DataParallel(net).cuda(0)

    # 定义数据加载
    train_dl = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=pin_memory)

    # 定义损失函数
    loss_func = DiceLoss()

    # 定义优化器
    opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)

    # 学习率衰减
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [900])

    # 训练网络
    start = time.time()
    for epoch in range(Epoch):
        lr_decay.step()
        mean_loss = []

        for step, (ct, seg) in enumerate(train_dl):

            ct = ct.to(device)  # .cuda()

            outputs_stage1, outputs_stage2 = net(ct)
            loss = loss_func(outputs_stage1, outputs_stage2, seg)

            mean_loss.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 4 == 0:
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min    {}'
                      .format(epoch, step, loss.item(), (time.time() - start) / 60, time.strftime('%Y-%m-%d %H:%M:%S')))

        mean_loss = sum(mean_loss) / len(mean_loss)

        # 每十个个epoch保存一次模型参数
        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        if epoch % 10 == 0:
            save_model(epoch, net, opt, opt, loss_func,
                       os.path.join(config.model_dir, f"net{epoch}-{loss.item():.3f}-{mean_loss:.3f}.pth"))
            # torch.save(net.state_dict(),
            #            os.path.join(config.model_path, f"net{epoch}-{loss.item():.3f}-{mean_loss:.3f}.pth"))
