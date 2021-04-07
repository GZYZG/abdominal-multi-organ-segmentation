"""
训练脚本
"""

import os
import time
import pandas as pd
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from config.config import config
from net.ResUnet_dice import net
from loss.ava_Dice_loss import DiceLoss
from data_prepare.dataset import train_ds
from utils.utils import *
import sys


if __name__ == "__main__":
    # 定义超参数
    on_server = True
    device = config.device
    print(f"{sys.argv[0]}\n\n")
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,list(range(torch.cuda.device_count()))))  # '0'  # if on_server is False else '1,2,3'
    cudnn.benchmark = True
    Epoch = config.epoch
    leaing_rate = 1e-4

    batch_size = config.batch_size
    num_workers = config.num_workers  # 1 if on_server is False else 1
    pin_memory = False if on_server is False else True

    # net = net.to(device)
    if config.on_gpu:
        gpu_ids = list(range(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        net = net.cuda(gpu_ids[0])

    # 定义数据加载
    shuffle = False
    train_dl = DataLoader(train_ds, batch_size, shuffle, num_workers=num_workers, pin_memory=pin_memory)

    # 定义损失函数
    loss_func = DiceLoss()

    # 定义优化器
    opt = torch.optim.Adam(net.parameters(), lr=leaing_rate)

    # 学习率衰减
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [900])

    if config.restore_training:
        checkpoint = load_model(net, opt, lr_decay, config.checkpoint_path)
        net, opt, lr_decay, save_epoch = checkpoint['model'], checkpoint['optimizer'], checkpoint['lr_schedule'], \
                                          checkpoint['epoch']
        start_epoch = save_epoch + 1
        print(f"{'*' * 5}\tRestore model from {config.checkpoint_path}...")
    else:
        start_epoch = 0
        print(f"{'*' * 5}\tTraining model from the start...")
    # 训练网络
    start = time.time()
    all_loss = []
    best_mean_loss = 1e10
    for epoch in range(start_epoch, Epoch):
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

        all_loss.append(mean_loss.copy())
        mean_loss = sum(mean_loss) / len(mean_loss)

        # 每十个个epoch保存一次模型参数
        # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
        if mean_loss < best_mean_loss:
            # best_mean_loss = mean_loss
            # if epoch > 1000:
                try:
                    best_mean_loss = mean_loss
                    save_model(epoch, net, opt, opt, loss_func,
                               os.path.join(config.model_dir, f"net{epoch}-{loss.item():.3f}-{mean_loss:.3f}-"
                                                              f"{config.CT_width}x{config.CT_height}.pth"))

                    mean_loss_save_path = os.path.join(config.output_dir, f"mean_loss-{config.CT_width}x{config.CT_height}.csv")
                    if os.path.exists(mean_loss_save_path):
                        saved = pd.read_csv(mean_loss_save_path)
                    else:
                        saved = pd.DataFrame(columns=['epoch', 'mean_loss'])
                    df = pd.DataFrame(data={'mean_loss': all_loss, 'epoch': list(range(start_epoch, epoch+1))},
                                      columns=['epoch', 'mean_loss'])

                    saved = pd.concat([saved, df], axis=0)
                    saved.to_csv(mean_loss_save_path, index=False)
                    all_loss.clear()
                    start_epoch = epoch + 1
                    # df.to_csv(os.path.join(config.output_dir, f"mean_loss-{time.strftime('%Y-%m-%d %H-%M-%S')} - "
                    #                                         f"{config.CT_width}x{config.CT_height}.csv"), index=False)
                except Exception as exp:
                    print(f"{'! ' * 10}\tError occurs while saving model to {config.model_dir}. Error info: {exp}")
                # torch.save(net.state_dict(),
                #            os.path.join(config.model_path, f"net{epoch}-{loss.item():.3f}-{mean_loss:.3f}.pth"))


