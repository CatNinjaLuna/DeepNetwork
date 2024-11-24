"""
Name: Carolina Li
Date: Nov/16/2024
File: task2_AnalyzeFirstLayer.py
Purpose: 
This script loads a pre-trained neural network model and visualizes the filters from the first convolutional layer.
It extracts the weights from the first convolutional layer and displays them in a 3x4 grid.
Unused subplots are hidden, and each filter is shown as an image to understand how the model detects features.
"""

import torch
import matplotlib.pyplot as plt
from task1 import MyNetwork

# Step 1: Load the Pre-trained Model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Ensure the model is in evaluation mode
# Step 2: Print Model Structure
print(model)
# Step 3: Extract Weights from the First Convolutional Layer
conv1_weights = model.conv1.weight.detach()  # Shape: [10, 1, 5, 5]
# Step 4: Visualize Filters
fig, axes = plt.subplots(3, 4, figsize=(12, 8))
fig.suptitle('Conv1 Filters Visualization', fontsize=16)

for i in range(conv1_weights.shape[0]):
   ax = axes[i // 4, i % 4]
   ax.imshow(conv1_weights[i, 0].numpy())
   ax.set_title(f'Filter {i+1}')
   ax.axis('off')
# Remove empty subplots (last row will have empty axes if filters < 12)
for i in range(conv1_weights.shape[0], 12):
   axes[i // 4, i % 4].axis('off')

plt.tight_layout()
plt.show()