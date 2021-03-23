import torch


def save_model(epoch, model, optimizer, loss, lr_schedule, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'lr_schedule': lr_schedule.state_dict()
    }, path)


def load_model(ModelNet, Optimizer, Lr_schedule, path):
    model = ModelNet()
    optimizer = Optimizer()

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    Lr_schedule.load_state_dict(checkpoint['lr_schedule'])

    return model, optimizer, loss, epoch, Lr_schedule