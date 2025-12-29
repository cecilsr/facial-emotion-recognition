# Comparing a Multilayer Perceptron with Convolutional Neural Networks for Facial Emotion Recognition

**Authors:**  
Benedetta Bruno, Vaitian L. Marianayagam, Cecilie S. Rønnestad & Sander S. Svartbekk

## Overview
This repository contains code and results for a machine learning project comparing a Multilayer Perceptron (MLP) and Convolutional Neural Networks (CNN) for facial emotion recognition. The work focuses on evaluating model performance and architectural differences on a real-world facial expression dataset.

A detailed report describing the theoretical background, methodology and results
is available in the `Report/` folder.

---

## Dataset
The project uses the **Real-world Affective Faces Database (RAF-DB)**.

The aligned version of the dataset (faces centered and resized to 100×100 pixels) was used.
Instructions for obtaining the dataset are available at:
http://www.whdeng.cn/RAF/model1.html

The file `Dataset/list_partition_label.txt` contains the predefined data split and emotion labels used in the experiments.


## Installation
Install required Python packages using:
```bash
pip install -r requirements.txt
