"""
Image transformations for facial emotion recognition.

Adapted from official pytorch tutorials.
"""



import torch
from torchvision import transforms


class ToTensorSample:
    """Convert image type to tensor and label to torch.long."""

    def __call__(self, sample):
        image, label = sample['image'], sample['emolabel']
        image = transforms.ToTensor()(image)  #  Scales to [0,1] and reshapes to CxHxW
        label = torch.tensor(label, dtype=torch.long)  #  Pytorch prefers this
        return {'image': image, 'emolabel': label}
    
class NormalizeSample:
    """Normalize image tensor using fixed mean and std."""
    
    def __init__(self):
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        self.std  = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    def __call__(self, sample):
        image, label = sample['image'], sample['emolabel']
        image = (image - self.mean) / self.std
        return {'image': image, 'emolabel': label}