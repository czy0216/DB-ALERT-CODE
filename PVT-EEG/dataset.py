import torch
from torch.utils.data import Dataset

class EEGData(Dataset):
    def __init__(self, x, y):
        super(EEGData, self).__init__()
        # Load EEG data and labels
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x_tensor = torch.tensor(self.x[index], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[index], dtype=torch.long)
        return x_tensor, y_tensor

    def __len__(self):
        return len(self.x)


