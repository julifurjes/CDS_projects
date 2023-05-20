# Assignment 3: Using pretrained CNNs for image classification

## Task description

In the previous assignments involving classification, we were performing a kind of simple feature extraction on images by making them greyscale and flattening them to a single vector. This vector of pixel values was then used as the input for some kind of classification model.

For this assignment, we're going to be working with an interesting kind of cultural phenomenon - fashion. On UCloud, you have access to a dataset of *Indo fashion* taken from this [Kaggle dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset). There is a paper which goes along with it on *arXiv.org*, which you can read [here](https://arxiv.org/abs/2104.02830).

Your instructions for this assignment are short and simple:

- You should write code which trains a classifier on this dataset using a *pretrained CNN like VGG16*
- Save the training and validation history plots
- Save the classification report

## Methods

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

1. Clone this reepository on your own device
2. Open Terminal at this folder or type ```cd assignment1-simple-image-search-julifurjes-main```
3. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
4. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment

## Results

You can find the classification report in the ```out``` folder, along with the training and validation history plots.
