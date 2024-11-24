'''
Name: Carolina Li
Date: Nov/16/2024
File: task1_Test_Handwritten_digits.py
Purpose:
This script loads a pre-trained neural network model to classify handwritten digits using the MNIST dataset format.
It preprocesses input images to match the MNIST specifications and predicts digit labels for a set of sample images.
The results are displayed in a 2x5 grid, showing the original images and their predicted classifications.
'''

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from task1 import MyNetwork  # Make sure task1.py has your MyNetwork class definition
import numpy as np

# Step 1: Load the Pre-trained Model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Set the model to evaluation mode

# Step 2: Define Image Preprocessing Transformations
transform = transforms.Compose([
   transforms.Grayscale(),          # Convert image to grayscale
   transforms.Resize((28, 28)),     # Resize to 28x28 pixels
   transforms.ToTensor(),           # Convert to PyTorch tensor
   transforms.Normalize((0.1307,), (0.3081,))  # Normalize to match MNIST dataset
])

# Step 3: Function to Load and Preprocess Images
# Step 3: Function to Load and Preprocess Images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    # Convert PIL image to NumPy array
    image_array = np.array(image)
    # Invert the image (since MNIST has white on black, but your input might have black on white)
    image_array = 255 - image_array
    # Convert back to PIL image
    image = Image.fromarray(image_array)
    image = transform(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
    return image

# Step 4: Test the Model on Handwritten Digits
def test_handwritten_digits():
   predictions = []
   images = []

   for digit in range(10):
      image_path = f'resized_digit_{digit}.png'
      try:
         input_image = preprocess_image(image_path)
         output = model(input_image)
         predicted_label = torch.argmax(output).item()
         
         predictions.append(predicted_label)
         images.append(input_image.squeeze().numpy())
            
         print(f"Digit {digit}: Predicted as {predicted_label}")
      except FileNotFoundError:
         print(f"Image {image_path} not found. Make sure all files are named correctly.")
   
   # Step 5: Display Results in a 2x5 Grid
   fig, axs = plt.subplots(2, 5, figsize=(12, 6))
   fig.suptitle('Handwritten Digits Classification', fontsize=16)
   for idx, ax in enumerate(axs.flatten()):
      if idx < len(images):
         ax.imshow(images[idx], cmap='gray')
         ax.set_title(f"Pred: {predictions[idx]}")
         ax.axis('off')
   plt.tight_layout()
   plt.show()

# Run the function
test_handwritten_digits()