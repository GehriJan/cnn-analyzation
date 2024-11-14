# initial code from https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80

import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from load import ImageDataset
from confusion_matrix import display_confustion_matrix_plot
import numpy as np

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.linearLayer1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linearLayer1(x)
        return x

def check_accuracy(loader, model, train):
    if train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    model.train()

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 784
    num_classes = 26
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 10

    csv_file_train = 'dataset/sign_mnist_train/sign_mnist_train.csv'
    csv_file_test = 'dataset/sign_mnist_test/sign_mnist_test.csv'
    dataset_train = ImageDataset(csv_file_train)
    dataset_test = ImageDataset(csv_file_test)

    # Create a DataLoader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

    model = CNN(in_channels=1, num_classes=num_classes).to(device)

    activactions = {}
    def get_activation(name):
        def hook(model, input, output):
            activactions[name] = output.detach()
        return hook
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.relu1.register_forward_hook(get_activation('relu1'))
    model.pool.register_forward_hook(get_activation('pool'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.relu2.register_forward_hook(get_activation('relu2'))
    model.linearLayer1.register_forward_hook(get_activation('linearLayer1'))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
            # Move data and targets to the device
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    check_accuracy(train_loader, model, train=True)
    check_accuracy(test_loader, model, train=False)

    display_confustion_matrix_plot(model, test_loader)