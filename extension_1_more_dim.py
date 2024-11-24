"""
Name: Carolina Li
Date: Nov/18/2024
File: extension_1_more_dim.py
Purpose: 
This code file applies convolutional filters from a pre-trained MyNetwork model 
to a sample MNIST image using PyTorch and OpenCV, visualizing the filters 
and their outputs. While functional, the visualization could be improved by
fixing a duplicate imshow call and enhancing subplot alignment.
We're going to recognize alpha, beta and gemma in this code file.
This code file using increased dimension in the MaxPool2D from [1, 10, 12, 12] to [1, 20, 4, 4]
"""


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from task1 import MyNetwork

# Step 1: Load Pre-trained MNIST Model
model = MyNetwork()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Replace the Last Layer for Greek Letter Classification
model.fc2 = nn.Linear(model.fc2.in_features, 3)  # 3 Greek letters: alpha, beta, gamma
print("Updated Model:")
print(model)

# Step 2: Define Transformations
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

transform = transforms.Compose([
    transforms.ToTensor(),
    GreekTransform(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Step 3: Prepare Datasets
training_set_path = 'greek_train'
greek_train = DataLoader(
    datasets.ImageFolder(training_set_path, transform=transform),
    batch_size=5,
    shuffle=True
)

hand_drawn_path = 'resized_greek'
hand_drawn_dataset = DataLoader(
    datasets.ImageFolder(hand_drawn_path, transform=transform),
    batch_size=1,
    shuffle=False
)

# Step 4: Inspect Layer Dimensions
def inspect_model_dimensions(model, input_tensor):
    outputs = {}
    def hook(module, input, output):
        outputs[module] = output.shape

    hooks = []
    for layer in model.children():
        hooks.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        model(input_tensor)

    for layer, shape in outputs.items():
        print(f"Layer: {layer}, Output Shape: {shape}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

# Prepare a dummy input to inspect layer dimensions
dummy_input = torch.randn(1, 1, 28, 28)
print("\nLayer-wise Output Dimensions:")
inspect_model_dimensions(model, dummy_input)

# Step 5: Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc2.parameters(), lr=0.001)

# Training Loop
epochs = 10
losses = []
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in greek_train:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(greek_train)
    losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Plot Training Loss
plt.plot(range(1, epochs + 1), losses, marker='o')
plt.title("Training Loss on Greek Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Step 6: Evaluation
def evaluate_model(model, data_loader, class_map):
    correct = 0
    total = 0
    per_class_correct = {key: 0 for key in class_map.values()}
    per_class_total = {key: 0 for key in class_map.values()}
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for label, pred in zip(labels, predicted):
                class_name = class_map[label.item()]
                per_class_total[class_name] += 1
                if pred.item() == label.item():
                    per_class_correct[class_name] += 1

    accuracy = correct / total * 100
    per_class_accuracy = {key: 100 * per_class_correct[key] / per_class_total[key]
                          for key in class_map.values()}
    return accuracy, per_class_accuracy

# Map indices to class names
class_map = {v: k for k, v in greek_train.dataset.class_to_idx.items()}

# Evaluate on hand-drawn dataset
accuracy, per_class_accuracy = evaluate_model(model, hand_drawn_dataset, class_map)
print(f"\nEvaluation Accuracy on Hand-Drawn Greek Letters: {accuracy:.2f}%")
print("Per-Class Accuracy:")
for class_name, acc in per_class_accuracy.items():
    print(f"  {class_name}: {acc:.2f}%")

# Analyze Output Dimensions for a Single Sample
with torch.no_grad():
    for images, labels in hand_drawn_dataset:
        print(f"Input Shape: {images.shape}")
        outputs = model(images)
        print(f"Output Logits Shape: {outputs.shape}")
        _, predicted = torch.max(outputs, 1)
        print(f"Predicted Class: {predicted.item()}, Actual Class: {labels.item()}")
        break  # Check only one example
