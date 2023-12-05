from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        if self.labels is not None:
            assert len(self.data) == len(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.data[idx]

        item = self.data[idx]
        label = self.labels[idx]
        return item, label
