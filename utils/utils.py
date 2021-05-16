import torch


def save_model(epoch, model, optimizer, loss_func, lr_schedule, path):
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_func,
            'lr_schedule': lr_schedule.state_dict()
        }, path)
        return True
    except Exception as exp:
        print(f"Error occurs while saving model to {path}. Error info: {exp}")
        return False


def load_model(model, path, optimizer=None, lr_schedule=None):
    """从checkpoint文件加载模型

    Parameters
    ----------
    model : nn.Module, 创建好的未加载参数的模型
    optimizer : 优化器，可选的
    lr_schedule : 学习率scheduler，可选的
    path : str, checkpoint文件的路径
    """
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        if lr_schedule is not None:
            lr_schedule.load_state_dict(checkpoint['lr_schedule'])

        return dict(zip(["model", "optimizer", "lr_schedule", "epoch"], [model, optimizer, lr_schedule, epoch]))
    except Exception as exp:
        print(f"Error occurs while load model from {path}. Error info: {exp}")
        raise RuntimeError(f"Load Model from {path} failed! Error info: {exp}")
        # return dict(zip(["model", "optimizer", "lr_schedule", "epoch"], [None, None, None, None]))
