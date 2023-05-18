# Assignment 3: Using pretrained CNNs for image classification

## About the assignment

In this assignment, I have trained a classifier on an Indo fashion dataset using a *pretrained CNN like VGG16*. The dataset is not included in the repository, due to its great size, but it can be downloaded from [here](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). It  consists of 106K images and 15 unique cloth categories. There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

NB: My code also includes the unzipping process, therefore if you have already unzipped it before uploading it, upload it in a folder called ´data´. This way the unzipping will be automatically skipped.

Furthermore, only limited datapoints were used for the training, due to time efficiency issues. However, this can be easily changed by removing the sampling when defining the datasets.

## File structure

- In the ```src``` folder you can find the script, in a ```.py``` format.
- The training and validation history plots and the classification report is saved in the ```out``` folder.
- The ```archive``` folder contains the older versions of the script.
- The ```setup.sh``` file includes everything to run the installments.
- The ```run.sh``` file includes everything to run the script.

## How to run it
First write ```bash setup.sh``` in the terminal to install the requirements and virtual environment, and run ```bash run.sh``` to run the script.