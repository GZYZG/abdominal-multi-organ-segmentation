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


def load_model(ModelNet, Optimizer, Lr_schedule, path):
    try:
        model = ModelNet()
        optimizer = Optimizer()

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        Lr_schedule.load_state_dict(checkpoint['lr_schedule'])

        return model, optimizer, loss, epoch, Lr_schedule
    except Exception as exp:
        print(f"Error occurs while load model from {path}. Error info: {exp}")
        return None, None, None, None, None
