import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load the CSV into a DataFrame
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        # Return the number of rows in the CSV (number of images)
        return len(self.data)

    def __getitem__(self, idx):
        # Get label and pixel data
        label = self.data.iloc[idx, 0]
        pixels = self.data.iloc[idx, 1:].values.astype(np.float32).reshape(28, 28)  # Reshape to 28x28

        # Normalize pixels to [0, 1]
        pixels /= 255.0

        # Convert to tensor
        image = torch.tensor(pixels).unsqueeze(0)  # Shape becomes [1, 28, 28]

        if self.transform:
            image = self.transform(image)

        return image, label
