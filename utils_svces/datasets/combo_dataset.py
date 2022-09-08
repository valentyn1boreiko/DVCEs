import torch
from torch.utils.data import Dataset

class ComboDataset(Dataset):
    def __init__(self, datasets):
        num_datasets = len(datasets)
        self.dataset_lengths = torch.zeros(num_datasets, dtype=torch.long)

        for i, ds in enumerate(datasets):
            self.dataset_lengths[i] = len(ds)

        self.cum_lengths = torch.cumsum(self.dataset_lengths, dim=0)
        self.length = torch.sum(self.dataset_lengths)
        self.datasets = datasets

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        ds_idx = torch.nonzero(self.cum_lengths > index, as_tuple=False)[0]
        if ds_idx > 0:
            item_idx = index - self.cum_lengths[ds_idx - 1]
        else:
            item_idx = index

        return self.datasets[ds_idx][item_idx]
