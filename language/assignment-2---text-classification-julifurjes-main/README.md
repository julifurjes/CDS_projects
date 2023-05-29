# Assignment 2: Text Classification

## About the assignment

In this assignment, I have extracted information from the Fake or Real News dataset and used ```scikit-learn``` to train simple (binary) classification models on it. This was performed by using three ```.py``` scripts:
  - ```vectorizer_file.py``` is responsible for the vectorizing and the training of the dataset
  - ```logistic_regression_2.0.py``` is responsible for creating a classifier via logistic regression
  - ```neural_network_2.0.py``` is responsible for creating a classifier via neural network

Finally, the trained models and the classification reports are saved for future analysis purposes.

This assignment is designed to test that you can:

1. Train simple benchmark machine learning classifiers on structured text data;
2. Produce understandable outputs and trained models which can be reused;
3. Save those results in a clear way which can be shared or used for future analysis

## File structure

- In the ```src``` folder you can find the above mentioned three scripts:
  - ```vectorizer_file.py```
  - ```logistic_regression_2.0.py```
  - ```neural_network_2.0.py```
- The classifiers are saved in the ```out``` folder.
- The models are saved in the ```models``` folder.
- There is an additional folder called ```archive``` where the old versions of the scripts are stored.
- The data is available in the ```data``` folder or via [this link](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news).
- The ```setup.sh``` file includes everything to run the installments.
- The ```run.sh``` file includes everything to run the script.

## How to run it
First write ```bash setup.sh``` in the terminal to install the requirements and virtual environment, and run ```bash run.sh``` to run the script.
=======

## How to run it

1. Clone this repository on your own device
2. Open Terminal at this folder or type ```cd assignment-2---text-classification-julifurjes```
3. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
4. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment

## Notes

Make sure that you have all the files in your cloned folder and that your ```data``` folder is not empty.

## Results

You can find the classifier reports from the models in the ```out``` folder.
