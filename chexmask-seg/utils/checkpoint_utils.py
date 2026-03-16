import torch
from os.path import join

def load_checkpoint(model, optimizer, log_path, save_type):
    """
    args:
        model: model to load model_state_dict from checkpoint
        optimizer: optimzer to load optimizer_state_dict from checkpoint
        load_path: path where all log files are saved   
        save_type : save type for the checkpoint - "best_epoch" or "latest_epoch"
    """
    w_path = 'weights/model-{}.pt'.format(save_type)
    checkpoint = torch.load(join(log_path, w_path))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, checkpoint['epoch'], checkpoint['metric_monitor']

def save_checkpoint(model, optimizer, epoch, metric_monitor, log_path, save_type):
    """
    params:
            epoch -- checkpoint epoch number
            save_type -- save type for the checkpoint - "best_epoch" or "latest_epoch"
    """
    w_path = 'weights/model-{}.pt'.format(save_type)
    torch.save({'epoch'               : epoch,
                'metric_monitor'      : metric_monitor,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                join(log_path, w_path))
    print(f"===> New {save_type} saved: Epoch--{epoch:03d}")


def get_model(model, log_path, save_type):
    """
    get best trained model -- useful for active learning acquisition

    returns:
            model -- Segmentation UNet weights for the lowest valid BCE
    """
    w_path = 'weights/model-{}.pt'.format(save_type)
    checkpoint = torch.load(join(log_path, w_path))

    model.load_state_dict(checkpoint['model_state_dict'])

    return model
