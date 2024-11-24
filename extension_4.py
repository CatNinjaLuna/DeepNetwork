"""
Name: Carolina Li
Date: Nov/19/2024
File: extension_4.py
Purpose: 
This code file demonstrates how to load a pre-trained ResNet-18 model and visualize the filters 
from its first convolutional layer to understand the learned feature representations. 
It extracts the filters, normalizes them for visualization, and plots them in a grid using Matplotlib.
"""


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Load Pre-trained ResNet-18
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()  # Ensure it's in evaluation mode
# Print Model Architecture to Identify Conv Layers
print(resnet18)

# Extract the First and Second Convolutional Layers
conv1 = resnet18.conv1  # First convolutional layer
conv2 = list(resnet18.layer1)[0].conv1  # First convolution in layer1
# Extract Filters from the First Convolutional Layer
conv1_filters = conv1.weight.detach().cpu().numpy()

# Plot Filters
def plot_filters(filters, num_columns=8):
   num_filters = filters.shape[0]
   num_rows = num_filters // num_columns + (num_filters % num_columns > 0)
   fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 15))
   for i in range(num_filters):
        ax = axes[i // num_columns, i % num_columns]
        # Normalize filter for visualization
        filt = filters[i]
        filt = (filt - filt.min()) / (filt.max() - filt.min())
        ax.imshow(filt.transpose(1, 2, 0))  # Transpose to (H, W, C)
        ax.axis('off')
        ax.set_title(f"Filter {i+1}")
   plt.tight_layout()
   plt.show()

# Visualize Conv1 Filters
plot_filters(conv1_filters)

