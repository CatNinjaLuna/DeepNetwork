"""
Name: Carolina Li
Date: Nov/18/2024
File: extension_2_Greek.py
Purpose: 
This code file applies convolutional filters from a pre-trained MyNetwork model 
to a sample MNIST image using PyTorch and OpenCV, visualizing the filters 
and their outputs. While functional, the visualization could be improved by
fixing a duplicate imshow call and enhancing subplot alignment.
We're going to recognize epsilon, omega, and rho in this code file.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from task1 import MyNetwork
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Step 1: Load Pre-trained MNIST Model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Step 2: Freeze All Layers
for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace the Last Layer (Adjust for 3 Greek Letters)
model.fc2 = nn.Linear(model.fc2.in_features, 3)  # 3 nodes for epsilon, omega, and rho
print(model)

# Greek Dataset Transform
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

# Compose Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    GreekTransform(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# DataLoader for Training and Evaluation
training_set_path = 'Greek_train_extension'
Greek_train_extension = DataLoader(
    datasets.ImageFolder(training_set_path, transform=transform),
    batch_size=5,
    shuffle=True
)

hand_drawn_path = 'extension_resized_greek'
hand_drawn_dataset = DataLoader(
    datasets.ImageFolder(hand_drawn_path, transform=transform),
    batch_size=1,
    shuffle=False
)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc2.parameters(), lr=0.001)

# Training Loop
epochs = 58
losses = []
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in Greek_train_extension:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(Greek_train_extension)
    losses.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
print("Training Complete!")

# Plot the loss
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title("Training Error on Greek Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Evaluation on Handwritten Greek Letters
model.eval()
predictions = []
actual = []

# Per-class Accuracy Tracking
class_counts = {0: 0, 1: 0, 2: 0}  # Total examples per class
class_correct = {0: 0, 1: 0, 2: 0}  # Correct predictions per class

with torch.no_grad():
    for images, labels in hand_drawn_dataset:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())
        actual.append(labels.item())
        # Track per-class statistics
        class_counts[labels.item()] += 1
        if predicted.item() == labels.item():
            class_correct[labels.item()] += 1

# Map indices to class names
class_map = {v: k for k, v in Greek_train_extension.dataset.class_to_idx.items()}

# Calculate Per-Class Accuracy
print("\nPer-Class Accuracy:")
for class_idx, total in class_counts.items():
    if total > 0:
        accuracy = (class_correct[class_idx] / total) * 100
        print(f"  {class_map[class_idx]}: {accuracy:.2f}% ({class_correct[class_idx]}/{total})")
    else:
        print(f"  {class_map[class_idx]}: No samples.")

# Overall Accuracy
total_correct = sum(class_correct.values())
total_samples = sum(class_counts.values())
overall_accuracy = (total_correct / total_samples) * 100
print(f"\nOverall Accuracy on Hand-Drawn Greek Letters: {overall_accuracy:.2f}%")

# Print Detailed Predictions
print("\nDetailed Predictions:")
for i, (pred, true) in enumerate(zip(predictions, actual)):
    print(f"Image {i + 1}: Predicted = {class_map[pred]}, Actual = {class_map[true]}")
