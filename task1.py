"""
Name: Carolina Li
Date: Nov/16/2024
File: task1.py
Purpose: The objective of this file is to build and train a network to recognize digits using the MNIST database.
Then we will visualize the first 6 digits from the test set using the matplotlib packages.
Lastly, we'll build a network model for digit recognition.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchsummary import summary

# Define the Neural Network
# Define the Neural Network 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Neural Network class
class MyNetwork(nn.Module):
   def __init__(self):
      super(MyNetwork, self).__init__()
      # Define layers as per the given instructions
      self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 1 input channel, 10 output channels, 5x5 filter
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with 2x2 window
      self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 10 input channels, 20 output channels, 5x5 filter
      self.dropout = nn.Dropout(0.5)  # Dropout layer with 50% dropout rate
      self.fc1 = nn.Linear(320, 50)  # Fully connected layer with 320 inputs to 50 nodes
      self.fc2 = nn.Linear(50, 10)   # Fully connected layer to 10 output classes

   # Perform a forward pass through the network, applying convolution, pooling, dropout, and fully connected layers.
   def forward(self, x):
      # Forward pass through the network
      x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
      x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
      x = self.dropout(x)                   # Apply Dropout
      x = x.view(-1, 320)                   # Flatten the tensor
      x = F.relu(self.fc1(x))               # Fully connected layer 1 -> ReLU
      x = self.fc2(x)                       # Fully connected layer 2 (output)
      return F.log_softmax(x, dim=1)        # Apply log_softmax for output

# Train the neural network model for a given number of epochs, tracking training and testing accuracy.
def train_network(model, train_loader, test_loader, epochs=5, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_acc_list, test_acc_list = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        correct_train, total_train = 0, 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(output, 1)
            correct_train += (predicted == target).sum().item()
            total_train += target.size(0)
        train_acc = correct_train / total_train
        train_acc_list.append(train_acc)

        # Evaluation phase
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output, 1)
                correct_test += (predicted == target).sum().item()
                total_test += target.size(0)
        test_acc = correct_test / total_test
        test_acc_list.append(test_acc)

        print(f'Epoch {epoch+1}/{epochs} | Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}')

    # Plotting the accuracy after training is complete
    plt.figure()
    plt.plot(range(1, epochs + 1), train_acc_list, label='Train Accuracy', color='blue')
    plt.plot(range(1, epochs + 1), test_acc_list, label='Test Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Testing Accuracy')
    plt.grid(True)
    plt.show()
            

# main function
def main(argv):
   # MNIST data loading and transformations
   transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))  # Normalizing as per MNIST dataset stats
   ])   
   train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

   # Plot first 6 test images
   examples = iter(test_loader)
   example_data, example_targets = next(examples)
   fig, axes = plt.subplots(1, 6, figsize=(10, 2))
   for i in range(6):
      axes[i].imshow(example_data[i][0], cmap='gray')
      axes[i].set_title(f'Label: {example_targets[i]}')
      axes[i].axis('off')
      # Save the entire figure as a single image
   plt.savefig('mnist_examples.png', bbox_inches='tight')  # Saves with tight layout
   plt.show()

   # Initialize and train the model
   model = MyNetwork()
   # Print a summary of the model
   summary(model, input_size=(1, 28, 28))
   train_network(model, train_loader, test_loader, epochs=5)

   # Save the trained model
   torch.save(model.state_dict(), 'mnist_model.pth')
   
   return

if __name__ == "__main__":
    main(sys.argv)