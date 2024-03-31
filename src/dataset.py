from torch.utils.data import Dataset
import torch

class NewsDataset(Dataset):
    def __init__(self, X, y, scaler):
        self.X = scaler.transform(X)
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return {
            "features": torch.tensor(self.X[index], dtype=torch.float32),
            "target": torch.tensor(self.y[index], dtype=torch.float32)
        }