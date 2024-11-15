# initial code from https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80

import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from load import ImageDataset
from xAI_methods import display_confustion_matrix_plot
import numpy as np
from model import CNN
from typing_extensions import Literal
from gradcam import perform_gradcam
from torchvision import transforms

# Data augmentation transformations
transform = transforms.Compose(
    [
        transforms.RandomRotation(10),  # Rotate images randomly by up to 10 degrees
        transforms.RandomHorizontalFlip(),  # Flip images horizontally with a probability of 0.5
        # transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),  # Minor translation and scaling
        transforms.RandomResizedCrop(
            28, scale=(0.9, 1.1)
        ),  # Crop and resize to add scale variation
    ]
)
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 784
    num_classes = 26
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 7
    load_model = False
    store_model = False
    model_path = "better_test_than_training.pth"

    xAI_method: Literal[
        "confusion_matrix",
        "grad-cam"
    ] = "grad-cam"

    csv_file_train = 'dataset/sign_mnist_train/sign_mnist_train.csv'
    csv_file_test = 'dataset/sign_mnist_test/sign_mnist_test.csv'
    dataset_train = ImageDataset(csv_file_train, transform=transform)
    dataset_test = ImageDataset(csv_file_test)

    # Create a DataLoader
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

    model = CNN(in_channels=1, num_classes=num_classes).to(device)

    if load_model == True:
        model.load_state_dict(torch.load("saved_models/" + model_path, map_location=device))
        print(f"Model: {model} loaded from file")
    else:
        print(f"Model: {model}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

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

    if xAI_method=="confusion_matrix":
        display_confustion_matrix_plot(model, test_loader)
    elif xAI_method=="grad-cam":
        perform_gradcam(model=model, dataset_test=dataset_test, device=device)
    if store_model:
        torch.save(model.state_dict(), "saved_models/" + model_path)