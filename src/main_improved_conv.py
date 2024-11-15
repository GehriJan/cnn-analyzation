# initial code from https://medium.com/@myringoleMLGOD/simple-convolutional-neural-network-cnn-for-dummies-in-pytorch-a-step-by-step-guide-6f4109f6df80

import random
import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from load import ImageDataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import to_pil_image
import numpy as np
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2  # Import OpenCV for resizing
from PIL import Image
import time
import os
from torchvision import transforms
from model import CNN

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = 784
    num_classes = 26
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 7
    load_model = False
    model_path = "tmp.pth"

    csv_file_train = "dataset/sign_mnist_train/sign_mnist_train.csv"
    csv_file_test = "dataset/sign_mnist_test/sign_mnist_test.csv"
    dataset_train = ImageDataset(csv_file_train, transform=transform)
    dataset_test = ImageDataset(csv_file_test)

    # Create a DataLoader
    train_loader = DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True
    )
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

    model.check_accuracy(train_loader, train=True)
    model.check_accuracy(test_loader, train=False)

    if load_model == False:
        torch.save(model.state_dict(), "saved_models/" + model_path)

    # Generate a unique timestamp
    timestamp = int(time.time())

    # Initialize Grad-CAM
    target_layer = model.conv4
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Prepare a test sample
    # sample_img, sample_label = dataset_test[random.randint(0, len(dataset_test))]
    sample_img, sample_label = dataset_test[27]
    sample_img = sample_img.unsqueeze(0).to(device)
    sample_img_pil = to_pil_image(sample_img[0].cpu())

    # Forward pass and prediction
    output = model(sample_img)
    predicted_class = output.argmax().item()

    folder_name = f"saved_images/{timestamp}_P{predicted_class}_L{sample_label}/"
    os.makedirs(folder_name, exist_ok=True)
    original_image_filename = f"{folder_name}original_image.png"
    sample_img_pil.save(original_image_filename)

    # Generate CAM
    grayscale_cam = cam(
        input_tensor=sample_img, targets=[ClassifierOutputTarget(predicted_class)]
    )
    grayscale_cam = grayscale_cam[0, :]

    # Convert the sample image to RGB by duplicating grayscale values across 3 channels
    rgb_image = np.array(to_pil_image(sample_img[0].cpu())).astype(np.float32) / 255.0
    rgb_image = np.stack([rgb_image] * 3, axis=-1)  # Convert grayscale to RGB

    # Resize the grayscale CAM to match the size of rgb_image
    grayscale_cam_resized = cv2.resize(
        grayscale_cam, (rgb_image.shape[1], rgb_image.shape[0])
    )

    # Expand grayscale CAM to 3 channels to match rgb_image
    grayscale_cam_resized_3ch = np.repeat(
        grayscale_cam_resized[:, :, np.newaxis], 3, axis=2
    )

    # Generate the overlay
    cam_image = show_cam_on_image(rgb_image, grayscale_cam_resized_3ch, use_rgb=True)

    # Convert cam_image to PIL for displaying
    cam_image_pil = to_pil_image(cam_image)
    # cam_image_pil.show()

    image_filename = f"{folder_name}cam_image.png"
    cam_image_pil.save(image_filename)