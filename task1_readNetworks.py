"""
Name: Carolina Li
Date: Nov/16/2024
File: task1_readNetworks.py
Purpose:
This script loads a pre-trained neural network model to evaluate its performance on the MNIST test dataset.
It iterates over the first 10 test samples, displaying the model's output values, predicted labels, and actual labels.
The results for the first 9 samples are visualized in a 3x3 grid, showing the input images along with their predicted and actual classifications.
"""

import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from task1 import MyNetwork  # Ensure task1.py contains your MyNetwork class definition

# Load the saved model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Set the model to evaluation mode

# Define the transformation for the MNIST dataset
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))  # Normalizing as per MNIST dataset stats
])   

# Load the MNIST test set
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
# Evaluate the model on the first 10 test samples
print("Evaluating the model on the first 10 test images:\n")
images = []
predictions = []
actual_labels = []

# iterate over first 10 examples
for i, (image, label) in enumerate(test_loader):
   # Run the model on the input image
   output = model(image)
   
   # Extract the output values, predicted label, and the actual label
   output_values = output.detach().numpy().flatten()
   predicted_label = torch.argmax(output).item()
   actual_label = label.item()

   # Print the 10 output values rounded to 2 decimal places
   output_str = ", ".join([f"{value:.2f}" for value in output_values])
   print(f"Image {i+1}:")
   print(f"Output values: [{output_str}]")
   print(f"Predicted: {predicted_label}, Actual: {actual_label}\n")
   # Collect data for visualization
   if i < 9:  # Store only the first 9 images for the plot
      images.append(image)
      predictions.append(predicted_label)
      actual_labels.append(actual_label)
   
   if i == 9:  # Stop after processing the first 10 images
      break

# Plotting the first 9 images in a 3x3 grid with their predictions
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle('MNIST Predictions on Test Set', fontsize=16)

for idx, ax in enumerate(axs.flatten()):
    if idx < len(images):
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.set_title(f"Pred: {predictions[idx]}, Actual: {actual_labels[idx]}")
        ax.axis('off')

plt.tight_layout()
plt.show()