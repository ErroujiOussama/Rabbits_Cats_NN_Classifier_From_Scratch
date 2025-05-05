
# Rabbits\_Cats\_NN\_Classifier\_From\_Scratch

This project implements a binary image classification neural network entirely from scratch using Python and NumPy. The model is designed to classify grayscale images of rabbits and cats based on raw pixel data, without the use of machine learning libraries.

## Description

The classification pipeline consists of the following components:

### 1. Data Preprocessing

* Images from two directories (`rabbit` and `cat`) are converted to grayscale and resized to 90×90 pixels.
* Each image is flattened into a 1D array of 8100 pixel intensity values.
* Data is labeled (0 for rabbit, 1 for cat), stored in a structured format using pandas, and normalized to the range \[0, 1].

### 2. Neural Network Architecture

* **Input Layer**: 8100 features (flattened pixel values).
* **Hidden Layer**: 128 neurons, activated using the ReLU function:

  $$
  \text{ReLU}(z) = \max(0, z)
  $$
 **Output Layer**: 2 neurons, activated using the softmax function for multi-class probability:

  $$
  \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
  $$

### 3. Forward Propagation

Given input $X$, parameters $W_1, B_1, W_2, B_2$, the network computes:

$$
Z_1 = W_1 X + B_1
$$

$$
A_1 = \text{ReLU}(Z_1)
$$

$$
Z_2 = W_2 A_1 + B_2
$$

$$
A_2 = \text{softmax}(Z_2)
$$

### 4. Backward Propagation

Gradients are computed manually using the chain rule. One-hot encoding is used for the labels. The gradients of the loss (cross-entropy) with respect to weights and biases are:

$$
dZ_2 = A_2 - Y_{\text{one-hot}}
$$

$$
dW_2 = \frac{1}{m} dZ_2 A_1^T,\quad dB_2 = \frac{1}{m} \sum dZ_2
$$

$$
dZ_1 = (W_2^T dZ_2) \cdot \mathbb{1}_{Z_1 > 0}
$$

$$
dW_1 = \frac{1}{m} dZ_1 X^T,\quad dB_1 = \frac{1}{m} \sum dZ_1
$$

### 5. Parameter Update

Using gradient descent with learning rate $\alpha$:

$$
W := W - \alpha \cdot dW,\quad B := B - \alpha \cdot dB
$$

### 6. Evaluation and Inference

The model is evaluated using classification accuracy. A sample image from the validation set is visualized and its predicted label is compared to the ground truth.

### 7. Model Saving

The final weights and preprocessing statistics (mean, standard deviation) are saved using `pickle`.
