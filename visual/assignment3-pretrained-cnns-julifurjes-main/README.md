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

The output called ```classification_report.txt``` gave the following report:

|                     | precision  |  recall | f1-score |  support|
| --- | --- | --- | --- | --- |
             | blouse      | 0.86     | 0.64     | 0.73       | 28|
       |  dhoti_pants      | 0.74     | 0.46     | 0.57       | 37|
       |     dupattas      | 0.41     | 0.50     | 0.45       | 24|
       |        gowns      | 0.52     | 0.37     | 0.43       | 30|
       |    kurta_men      | 0.40     | 0.68     | 0.51       | 31|
| leggings_and_salwars     | 0.71     | 0.59     | 0.65       | 37|
            | lehenga      | 0.76     | 0.70     | 0.73       | 27|
        | mojaris_men      | 0.73     | 0.73     | 0.73       | 33|
     |  mojaris_women      | 0.71     | 0.67     | 0.69       | 36|
      | nehru_jackets      | 0.65     | 0.74     | 0.69       | 27|
         |   palazzos      | 0.84     | 0.64     | 0.73       | 42|
         | petticoats      | 0.64     | 0.69     | 0.67       | 42|
          |     saree      | 0.56     | 0.91     | 0.69       | 33|
         |  sherwanis      | 0.64     | 0.16     | 0.25       | 45|
       |  women_kurta      | 0.34     | 0.71     | 0.47       | 28|

       |     accuracy   |           |         |    0.60   |    500|
       |    macro avg   |    0.63   |   0.61  |    0.60   |    500|
       | weighted avg   |    0.64   |   0.60  |    0.60   |    500|
