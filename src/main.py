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
from model import CNN

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 784
    num_classes = 26
    batch_size = 64
    learning_rate = 0.01
    num_epochs = 10
    load_model = False
    model_path = "tmp.pth"

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

    if load_model == True:
        model.load_state_dict(torch.load("saved_models/" + model_path, map_location=device))
        print(f"Model: {model} loaded from file")
    else:
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
    model.check_accuracy(train_loader, train=True, device=device)
    model.check_accuracy(test_loader, train=False, device=device)

    display_confustion_matrix_plot(model, test_loader)