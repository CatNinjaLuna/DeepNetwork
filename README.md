## Overview

Name: Carolina li
Date: Nov/8/2024
Project: Recgniton using Deep Networks

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

### Extension and Travel Days

I used the 3 travel days, and completed the following extensions:

Extension 1: Evaluate more dimensions on “Transfer Learning on Greek Letters”
Extension 2: train on more greek letters: omega, epsilon and rho
