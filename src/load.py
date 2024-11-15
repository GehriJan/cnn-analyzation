import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# # Load the dataset
# csv_file = 'dataset/sign_mnist_train/sign_mnist_train.csv'
# dataset = ImageDataset(csv_file)

# # Create a DataLoader
# train_data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# labels_map = {
#     0: "A",
#     1: "B",
#     2: "C",
#     3: "D",
#     4: "E",
#     5: "F",
#     6: "G",
#     7: "H",
#     8: "I",
#     9: "J",
#     10: "K",
#     11: "L",
#     12: "M",
#     13: "N",
#     14: "O",
#     15: "P",
#     16: "Q",
#     17: "R",
#     18: "S",
#     19: "T",
#     20: "U",
#     21: "V",
#     22: "W",
#     23: "X",
#     24: "Y",
#     25: "Z",
# }

# train_features, train_labels = next(iter(train_data_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")