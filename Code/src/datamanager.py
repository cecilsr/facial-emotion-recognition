"""PyTorch dataset for facial emotion recognition images.

Adapted from official pytorch tutorials: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html"""

import os
import torch
from skimage import io
from torch.utils.data import Dataset


class DataManager(Dataset):
    """Dataset for loading facial-emotion images and labels.

    Parameters
    ----------
    labels_df : pandas.DataFrame
        Dataframe with image filenames (col 0) and labels (col 1).
    root_dir : str
        Root directory containing images.
    transform : callable, optional
        Optional transform applied to each sample.
    """

    def __init__(self, labels_df, root_dir, transform=None):
        """
        Arguments:
            labels_df (pd dataframe): dataframe with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = labels_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        """Get a sample by index.

        Parameters
        ----------
        idx : int or torch.Tensor
            Sample index.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'image' : numpy.ndarray
            - 'emolabel' : int
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_df.iloc[idx, 0])
        image = io.imread(img_name)
        emolabel = self.labels_df.iloc[idx, 1] - 1
        sample = {'image': image, 'emolabel': emolabel}

        if self.transform:
            sample = self.transform(sample)

        return sample