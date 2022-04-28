# Multi-layer Artificial Neural Network ANN
1. This project implements a configurable multilayer neural network wrote in python. It is about a classifier based on machine learning.
2. The algorithm is based on early stopping criteria for the data pre-processing phase. Finally, it evaluates the accuracy on test dataset.
3. The network is configurable as a <b>deep</b> neural network or as a <b>shallow</b> network.
4. The dataset contains 60,000 grayscale images of size 28 x 28 pixels. Each image represents a handwritten digit from 0 to 9.

## Training
1. <b>Batch</b>: whole dataset is presented and then comes updated the network. 
2. <b>Online</b>: examines a data at a time before updating the network.
3. <b>Mini-Batch</b>: the dataset is divided into k batches processed in online mode. 

## Stopping Criteria
1. Stop as soon as the generalization loss exceeds a certain threshold. 
2. Use the quotient of generalization loss and progress.

## Dataset
1. Training: 50%
2. Validation: 25%
3. Test: 25%

For each training epoch, the training set and the validation set are randomly mixed.

## Activation Functions
1. Sigmoid (Standard logistic function) 
2. Tanh (Hyperbolic tangent)
3. Identity

## Output 
1. Training, Validation, Test error
2. Accuracy
3. F1
4. Precision
5. Recall
6. Confusion matrix

# Prerequisites
1. Python3
2. numpy
 ```bash
pip install numpy
```
4. mlxtend
 ```bash
pip install mlxtend
```
