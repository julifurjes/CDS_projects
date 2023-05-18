import numpy as np
import cv2
from tensorflow.keras.datasets import cifar10

def loading_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return X_train, y_train, X_test, y_test

def preprocessing(X_train_data, X_test_data):
    # convert to grayscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train_data])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test_data])
    # rescale
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    # reshaping
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    labels = ["airplane", 
          "automobile", 
          "bird", 
          "cat", 
          "deer", 
          "dog", 
          "frog", 
          "horse", 
          "ship", 
          "truck"]
    return X_train_dataset, X_test_dataset, labels

def main():
    X_train, y_train, X_test, y_test = loading_data()
    X_train_dataset, X_test_dataset, labels = preprocessing(X_train, X_test)
    return X_train, y_train, X_test, y_test, X_train_dataset, X_test_dataset, labels

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, X_train_dataset, X_test_dataset, labels = main()