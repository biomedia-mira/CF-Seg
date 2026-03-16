import numpy as np

def percentile_clip(img_numpy, min_val=0.5, max_val=99.5):
    """
    Intensity normalization based on percentile
    Clips the range based on the quarile values.
    min_val: should be in the range [0,100]
    max_val: should be in the range [0,100]
    """
    low = np.percentile(img_numpy, min_val)
    high = np.percentile(img_numpy, max_val)
    img_numpy[img_numpy < low] = low
    img_numpy[img_numpy > high] = high
    return img_numpy


def value_clip(img_numpy, min_val= -1024.0, max_val= 1024.0):
    """
    Intensity normalization based on exact value
    """
    img_numpy[img_numpy < min_val] = min_val
    img_numpy[img_numpy > max_val] = max_val
    return img_numpy

def normalize_intensity(img_tensor, clip_type, clip_value, normalization=None):
    """
    Accept the image tensor and normalizes it (ref: MedicalZooPytorch)
    Args: 
        img_tensor (tensor): image tensor
        normalization (string): choices = "max", "mean"
        norm_values (array, optional): (MEAN, STD, MAX, MIN)
    """
    if clip_type == "value":
        img_tensor = value_clip(img_tensor, clip_value[0], clip_value[1])
    else:
        img_tensor = percentile_clip(img_tensor, clip_value[0], clip_value[1])

    if normalization == "mean_percentile":
        mask = img_tensor > np.percentile(img_tensor, 1)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / std_val

    elif normalization == "max":
        img_tensor = img_tensor/img_tensor.max()

    elif normalization == 'full_volume_mean':
        img_tensor = (img_tensor - img_tensor.min()) / img_tensor.max()

    elif normalization == 'max_min':
        img_tensor -= img_tensor.min()
        img_tensor /= img_tensor.max()

    elif normalization == None:
        img_tensor = img_tensor

    return img_tensor


def pad_array(X, desired_shape):
    
    # Calculate the amount of padding needed for each axis
    pad_width = [(0, max(0, desired_shape[i] - X.shape[i])) for i in range(len(desired_shape))]
    
    # Pad the array with zeros
    padded_array = np.pad(X, pad_width, mode='constant')
    
    return padded_array
