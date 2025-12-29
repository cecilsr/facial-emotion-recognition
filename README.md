# Comparing a Multilayer Perceptron with Convolutional Neural Networks for Facial Emotion Recognition

**Authors:**  
Benedetta Bruno, Vaitian L. Marianayagam, Cecilie S. Rønnestad & Sander S. Svartbekk

## Overview
This repository contains code and results for a machine learning project comparing a Multilayer Perceptron (MLP) with Convolutional Neural Networks (CNN) for facial emotion recognition. The work focuses on evaluating model performance and architectural differences on a real-world facial expression dataset.

The CNN model achieved the best performance with a test accuracy of 76.60%, followed by a
CNN with Squeeze-and-Excitation blocks (74.22%) and the MLP (68.87%).

A detailed report describing the theoretical background, methodology and results
is available in the `Report` folder.


## Dataset
The project uses the Real-world Affective Faces Database (RAF-DB).

The aligned version of the dataset (faces centered and resized to 100×100 pixels) was used.
Instructions for obtaining the dataset are available at:
http://www.whdeng.cn/RAF/model1.html

The file `Dataset/list_partition_label.txt` contains the predefined data split and emotion labels used in the experiments.


## Installation
Install required Python packages using:
```bash
pip install -r requirements.txt
```

## Reproducibility
The results presented in the report can be reproduced by downloading the dataset from
the original source, placing the aligned images in the `Dataset` directory, and running
the provided training and evaluation scripts in the `Code` directory.


