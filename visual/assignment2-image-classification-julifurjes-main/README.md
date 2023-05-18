# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

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
First write ```bash setup.sh``` in the terminal to install the requirements and virtual environment, and run ```bash run.sh``` to run the script.