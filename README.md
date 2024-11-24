## Overview

Name: Carolina li
Date: Nov/8/2024
Project: Recgniton using Deep Networks
This project focuses on building, training, and analyzing deep networks for image recognition, using the MNIST digit recognition dataset as a foundation. It involves designing a convolutional neural network with layers for feature extraction, dropout for regularization, and fully connected layers for classification. Tasks include training the network, visualizing performance through training and testing accuracy plots, and saving the model for reuse. A key objective is to understand how network architecture and hyperparameters affect performance.
The project also includes transfer learning, where the trained MNIST network is adapted to classify Greek letters (alpha, beta, gamma, omega, epsilon, and rho) by freezing pre-trained layers and fine-tuning the final classification layer. Additional tasks explore visualizing convolutional filters and applying the trained model to real-world handwritten inputs, demonstrating the network’s ability to generalize.
A major component is experimentation, where network dimensions (e.g., number of layers, filter sizes, dropout rates) are varied to evaluate their impact on accuracy and training time. Extensions include recognizing additional Greek letters, applying pre-trained networks to other datasets, and testing innovative ideas like live video recognition. This project provides a comprehensive introduction to deep learning, offering hands-on experience in designing, fine-tuning, and evaluating neural networks.


## Ooperating System and IDE

MacOS (Apple M1 Pro chip) and Visual Studio Code(1.95.1 Universal)

## Commands for Installation

python -m venv venv
source venv/bin/activate
pip install torch torchvision matplotlib

## Commands for Compilation

Simply click on the upper right triangular icon to run the code

## Rescale the image

1. Install Image... package using XX command
2. This command is used to rescale the image:
   for file in \*.png; do
   convert "$file" -resize 128x128! "resized_$file"
   done

## Link to additional examples or video

Link to additional hand written examples:
https://drive.google.com/drive/folders/1yzTPuaHxROTmYMU16ZzvPZ9koftkK3pM?usp=sharing

Link to source of dataset of additional hand written greek letters:
https://www.kaggle.com/datasets/sayangupta001/mnist-greek-letters

link to video running the program:
Video of running the experiment: https://drive.google.com/file/d/1RdobQXe3io0j9_IEH20PldntTkNLhP-S/view?usp=drive_link
Video of running all tasks and extensions(without experiment):
https://drive.google.com/file/d/1VyfV-CqQEZ2OGyxJfT5Bw9FQ-01m-1BT/view?usp=sharing

## Acknowledgements

Pytorch tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
MNIST digit dataset tutorials: https://nextjournal.com/gkoehler/pytorch-mnist

## Reflections

Working on this task gave me a hands-on understanding of building and training a convolutional neural network (CNN) using PyTorch to recognize digits from the MNIST dataset. I learned how to design a model by stacking layers like convolution, pooling, dropout, and activation functions, and how to optimize its performance through iterative training and evaluation. Visualizing the filters from the first convolutional layer and seeing how they process images helped me understand how the network extracts features from input data. I also explored the concept of transfer learning by adapting the pre-trained model to recognize Greek letters, which showed me how to reuse existing models for new tasks with minimal adjustments. Overall, this project deepened my knowledge of CNNs, their flexibility, and the importance of visualization and testing in machine learning workflows.

### Extension and Travel Days

I used the 3 travel days, and completed the following extensions:

Extension 1: Evaluate more dimensions on “Transfer Learning on Greek Letters”
Extension 2: train on more greek letters: omega, epsilon and rho
Extension 4: There are many pre-trained networks available in the PyTorch package. Try loading one and evaluate its first couple of convolutional layers as in task 2.
