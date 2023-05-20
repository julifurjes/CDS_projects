# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## Task description

For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

The purpose of this assignment is:

- To ensure that you can use ```scikit-learn``` to build simple benchmark classifiers on image classification data
- To demonstrate that you can build reproducible pipelines for machine learning projects
- To make sure that you can structure repos appropriately

## Methods

In this assignment, I have used ```OpenCV``` to design a simple image search algorithm. First I have randomly chosen an image I'd like to work with. Then a colour histogram was extracted using ```OpenCV``` for that imgae first and then for the rest of the images. Then histogramss were compared and the five most similar images were chosen.

In this assignment, I have classified the ```Cifar10``` dataset by first loading and preprocessing the dataset. Then, the trained models and the classification reports are saved for future analysis purposes. This was performed by using three ```.py``` scripts:
  - ```vectorizer_file.py``` is responsible for the vectorizing and the training of the dataset
  - ```logistic_regression_2.0.py``` is responsible for creating a classifier via logistic regression
  - ```neural_network_2.0.py``` is responsible for creating a classifier via neural network

## File structure

- In the ```src``` folder you can find the above mentioned three scripts:
  - ```vectorizer_file.py```
  - ```log_regression.py```
  - ```neural_network.py```
- The classifiers are saved in the ```out``` folder.
- The models are saved in the ```models``` folder.
- The vectorizer script is stored in the ```utils``` folder.
- The ```setup.sh``` file includes everything to run the installments.
- The ```run.sh``` file includes everything to run the script.

## How to run it

1. Clone this reepository on your own device
2. Open Terminal at this folder or type ```cd assignment2-image-classification-julifurjes-main```
3. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
4. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment
