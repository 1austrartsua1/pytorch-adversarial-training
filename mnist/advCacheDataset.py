from torch.utils.data import Dataset
import torch 

class AdversarialCache(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset):
        """
        Args:
        dataset: pytorch dataset - the regular training set
        on which we will create adversaries, 1 for each training example
        """

        shape = list(dataset[0][0].shape)
        shape = [len(dataset)] + shape 
        self.advExamples = torch.zeros(shape)
        self.dataset = dataset


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        adv_example = self.advExamples[idx]
        real_example = list(self.dataset[idx])
        real_example.append(adv_example)
        return tuple(real_example)