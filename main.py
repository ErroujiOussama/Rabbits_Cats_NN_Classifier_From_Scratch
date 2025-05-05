import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os
import cv2


### preprocessing
def images_to_dataframe(cat_folder, rabbit_folder, image_size=(90, 90)):
    data = []
    labels = []
    
    # Helper function to load images
    def load_images_from_folder(folder, label):
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, image_size)  # Resize to match dataset style
                img_flattened = img_resized.flatten()
                data.append(img_flattened)
                labels.append(label)
    
    load_images_from_folder(rabbit_folder, label=0)  # Label 0 for rabbit
    load_images_from_folder(cat_folder, label=1)     # Label 1 for cat
    
    df = pd.DataFrame(data)
    df.insert(0, 'label', labels)
    
    return df

#df = images_to_dataframe('training_cats_rabbits/cat', 'training_cats_rabbits/rabbit/')
#df.to_csv('cat_rabbit_pixels.csv', index=False)


#### 
data = pd.read_csv('cat_rabbit_pixels.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

print(f'{m} {n}')

### 
train_data = data[0:int(0.8*m), :]
val_data = data[int(0.8*m):m, :]

X_train = train_data[:, 1:].T
X_train = X_train / 255.0
Y_train = train_data[:, 0]

X_val = val_data[:, 1:].T
X_val = X_val / 255.0
Y_val = val_data[:, 0]

###
print(X_val.shape)
print(Y_val.shape)
print(X_train.shape)
print(Y_train.shape)

def initialize_parameters():
  W1 = np.random.rand(128, 8100) - 0.5
  B1 = np.random.rand(128, 1) - 0.5
  W2 = np.random.rand(2, 128) - 0.5
  B2 = np.random.rand(2, 1) - 0.5
  return W1, B1, W2, B2

def ReLU(X):
  return np.maximum(X, 0)

# def softmax_calculator(Z):
#   return np.exp(Z) / sum(np.exp(Z))
def softmax_calculator(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # For numerical stability
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(W1, B1, W2, B2, X):
  Z1 = W1.dot(X) + B1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + B2
  A2 = softmax_calculator(Z2)
  return Z1, A1, Z2, A2

def one_hot_converter(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y] = 1
  return one_hot_Y.T

def backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
  one_hot_Y = one_hot_converter(Y)
  dZ2 = A2 - one_hot_Y
  dW2 = 1 / m * dZ2.dot(A1.T)
  dB2 = 1 / m * np.sum(dZ2)
  dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
  dW1 = 1 / m * dZ1.dot(X.T)
  dB1 = 1 / m * np.sum(dZ1)
  return dW1, dB1, dW2, dB2

def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
  W1 = W1 - learning_rate * dW1
  B1 = B1 - learning_rate * dB1
  W2 = W2 - learning_rate * dW2
  B2 = B2 - learning_rate * dB2
  return W1, B1, W2, B2

def get_predictions(A2):
  return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
  return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
  W1, B1, W2, B2 = initialize_parameters()

  for i in range(iterations):
    Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X)
    dW1, dB1, dW2, dB2 = backward_propagation(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)
    W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)

    if (i%20)==0:
      print("Iteration number: ", i)
      print("Accuracy = ", get_accuracy(get_predictions(A2), Y))
  return W1, B1, W2, B2

####################
W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 0.01, 1000)

##################
val_index = 1
Z1val, A1val, Z2val, A2val = forward_propagation(W1, B1, W2, B2, X_val[:, val_index, None])
print("Predicted label: ", get_predictions(A2val))
print("Actual label: ", Y_val[val_index])

image_array = X_val[:,val_index].reshape(90,90)
plt.imshow(image_array, cmap='gray')
plt.show()


Z1val, A1val, Z2val, A2val = forward_propagation(W1, B1, W2, B2, X_val)
val_acc = get_accuracy(get_predictions(A2val), Y_val)
print("Validation accuracy = ", val_acc)


##################
# Recompute training mean and std 
mean = np.mean(X_train, axis=1, keepdims=True)
std = np.std(X_train, axis=1, keepdims=True) + 1e-8

# Save model and preprocessing stats
import pickle
with open("trained_model.pkl", "wb") as f:
    pickle.dump((W1, B1, W2, B2, mean, std), f)

print("Model and normalization stats saved to 'trained_model.pkl'")


