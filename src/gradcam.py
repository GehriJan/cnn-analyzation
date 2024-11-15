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
from model import CNN

def perform_gradcam(model, dataset_test, device):
    # Generate a unique timestamp
    timestamp = int(time.time())

    # Initialize Grad-CAM
    target_layer = model.conv2
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
    cam_image_pil.show()

    image_filename = f"{folder_name}cam_image.png"
    cam_image_pil.save(image_filename)