'''
Name: Carolina Li
Date: Nov/18/2024
File: experiment.py
Purpose:
This code file aims to evaluate the impact of varying key architectural dimensions 
of a convolutional neural network (CNN) on its performance and training efficiency using the MNIST dataset. 
It automates the process of training and evaluating multiple configurations, storing results for analysis.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
from sklearn.metrics import accuracy_score

# Define Hyperparameter Grid
conv_layers_options = [2, 3, 4]
filters_options = [(16, 32), (32, 64), (64, 128)]
dropout_options = [0.3, 0.5, 0.7]
batch_size = 64
epochs = 5  # Adjust epochs for quicker experimentation

# Define Device (No GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyModifiedNetwork(nn.Module):
    def __init__(self, num_conv_layers, filters, dropout_rate):
        super(MyModifiedNetwork, self).__init__()
        layers = []
        in_channels = 1
        for i in range(num_conv_layers):
            out_channels = filters[i] if i < len(filters) else filters[-1]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        # Calculate the flattened size dynamically
        self.flatten_size = self._get_flatten_size(1, 1, 28, 28)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)

    def _get_flatten_size(self, batch_size, in_channels, height, width):
        # Pass a dummy input through the conv layers to calculate the output size
        dummy_input = torch.zeros(batch_size, in_channels, height, width)
        output = self.conv(dummy_input)
        return output.view(batch_size, -1).shape[1]

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.fc1(x))
        return self.fc2(x)
    
# Data Transformation
transform = transforms.Compose([
      transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST Data
mnist_train = DataLoader(datasets.MNIST('./data', train=True, transform=transform, download=True), 
                        batch_size=batch_size, shuffle=True)

mnist_test = DataLoader(datasets.MNIST('./data', train=False, transform=transform, download=True), 
                     batch_size=batch_size, shuffle=False)  

# Define a function to save results to a text file
def save_results_to_file(results, filename="results.txt"):
   with open(filename, "w") as file:
      file.write("Model Training Results\n")
      file.write("=" * 50 + "\n\n")
      for idx, result in enumerate(results, 1):
            file.write(f"Model Combination {idx}\n")
            file.write("-" * 50 + "\n")
            file.write(f"Conv Layers: {result['conv_layers']}\n")
            file.write(f"Filters: {result['filters']}\n")
            file.write(f"Dropout Rate: {result['dropout']}\n")
            file.write(f"Accuracy: {result['accuracy']:.4f}\n")
            file.write("\n" + "-" * 50 + "\n\n")
      # Write the best model configuration at the end
      best_model = max(results, key=lambda x: x['accuracy'])
      file.write("Best Model Configuration\n")
      file.write("=" * 50 + "\n")
      file.write(f"Conv Layers: {best_model['conv_layers']}\n")
      file.write(f"Filters: {best_model['filters']}\n")
      file.write(f"Dropout Rate: {best_model['dropout']}\n")
      file.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
      file.write("=" * 50 + "\n")

# Training and Evaluation Loop
results = []
for conv_layers, filters, dropout in itertools.product(conv_layers_options, filters_options, dropout_options):
   print(f"Training Model: ConvLayers={conv_layers}, Filters={filters}, Dropout={dropout}")
   
   # Initialize Model, Loss Function, and Optimizer
   model = MyModifiedNetwork(conv_layers, filters, dropout).to(device)
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   criterion = nn.CrossEntropyLoss()

   # Training Phase
   for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      for images, labels in mnist_train:
         images, labels = images.to(device), labels.to(device)
         
         optimizer.zero_grad()
         outputs = model(images)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()
         running_loss += loss.item()

      print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(mnist_train):.4f}")

   # Evaluation Phase
   model.eval()
   y_true = []
   y_pred = []

   with torch.no_grad():
      for images, labels in mnist_test:
         images, labels = images.to(device), labels.to(device)
         outputs = model(images)
         _, predicted = torch.max(outputs, 1)
         y_true.extend(labels.cpu().numpy())
         y_pred.extend(predicted.cpu().numpy())
   
   accuracy = accuracy_score(y_true, y_pred)
   print(f"Model Accuracy: {accuracy:.4f}")  

   # Store Results
   results.append({
      'conv_layers': conv_layers,
      'filters': filters,
      'dropout': dropout,
      'accuracy': accuracy
   })

   # Save Results to a File
   save_results_to_file(results, "experiement_model_results.txt")
   print("\nResults saved to 'model_results.txt'")



   # Print Summary of Results
   print("\nSummary of Results:")
   for result in results:
      print(f"ConvLayers={result['conv_layers']}, Filters={result['filters']}, Dropout={result['dropout']}, Accuracy={result['accuracy']:.4f}")

   # Sort and Display Best Model
   best_model = max(results, key=lambda x: x['accuracy'])
   print("\nBest Model Configuration:")
   print(f"ConvLayers={best_model['conv_layers']}, Filters={best_model['filters']}, Dropout={best_model['dropout']}, Accuracy={best_model['accuracy']:.4f}")

# Print Experiment Hyperparameters and Model Configuration
print("\nExperiment Configuration:")
print("=" * 50)
print(f"The number of convolution layers: {conv_layers_options}")
print(f"The size of the convolution filters: 3x3")
print(f"The number of convolution filters in a layer: {filters_options}")
print(f"The number of hidden nodes in the Dense layer: 128")
print(f"The dropout rates of the Dropout layer: {dropout_options}")
print(f"Whether to add another dropout layer after the fully connected layer: No, only one dropout layer is used after the Dense layer.")
print(f"The size of the pooling layer filters: 2x2")
print(f"The number or location of pooling layers: One pooling layer is added after each convolutional layer.")
print(f"The activation function for each layer: ReLU")
print(f"The number of epochs of training: {epochs}")
print(f"The batch size while training: {batch_size}")
print("=" * 50)
