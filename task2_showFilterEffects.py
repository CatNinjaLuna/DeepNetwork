"""
Name: Carolina Li
Date: Nov/16/2024
File: task2_showFilterEffects.py
Purpose: 
This code file applies convolutional filters from a pre-trained MyNetwork model to 
a sample MNIST image using PyTorch and OpenCV, visualizing the filters and 
their outputs in a grid. It effectively combines deep learning and image processing but 
could be improved by fixing duplicate imshow calls in the visualization.
"""
import torch
import cv2
from torchvision import datasets, transforms
from task1 import MyNetwork
import matplotlib.pyplot as plt
# Step 1: Load the Pre-trained Model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Step 2: Load a Sample Image from MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
sample_image, _ = mnist_train[0]
sample_image_np = sample_image.squeeze().numpy()

# Step 3: Extract the Filters from conv1
conv1_weights = model.conv1.weight.detach().numpy()

# Step 4: Apply Each Filter Using OpenCV's filter2D
filtered_images = []
with torch.no_grad():
   for i in range(conv1_weights.shape[0]):
      # Apply filter2D using OpenCV
      filtered_image = cv2.filter2D(sample_image_np, -1, conv1_weights[i, 0])
      filtered_images.append(filtered_image)
# Step 5: Visualize the Filtered Outputs
fig, axes = plt.subplots(5, 4, figsize=(10, 12))
fig.suptitle('Filtered Outputs', fontsize=16)

'''
for i in range(10):  # We have 10 filters and filtered images
   # First column: Display the filter itself
   ax1 = axes[i//2, i%2*2]  # 5 rows, place filters in the first column
   ax1.imshow(conv1_weights[i, 0], cmap='gray')  # Display filter as image
   ax1.set_title(f'Filter {i+1}')
   ax1.axis('off')

   # Second column: Display filtered image
   ax2 = axes[i//2, (1%2*2)+1]  # 5 rows, place filtered images in the second column
   ax2.imshow(filtered_images[i], cmap='gray')  # Display the result of ap
   ax2.set_title(f'Filtered Image {i+1}')
   ax2.axis('off')
'''
for i in range(10):  # We have 10 filters and filtered images
   # First column for filter
   ax1 = axes[i, 0] if i < 5 else axes[i - 5, 2]  # 1st column for first 5 filters, 3rd for next 5
   ax1.imshow(conv1_weights[i, 0], cmap='gray')  # Display filter as image
   ax1.set_title(f'Filter {i+1}')
   ax1.axis('off')

   # Second column for filtered image
   ax2 = axes[i, 1] if i < 5 else axes[i - 5, 3]  # 2nd column for first 5 images, 4th for next 5
   ax2.imshow(filtered_images[i], cmap='gray')
   ax2.imshow(filtered_images[i], cmap='gray')
   ax2.set_title(f'Filtered Image {i+1}')
   ax2.axis('off')



plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

