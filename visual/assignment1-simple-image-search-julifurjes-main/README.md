# Assignment 1: Building a simple image search algorithm

## Task description
For this assignment, you'll be using OpenCV to design a simple image search algorithm.

The dataset is a collection of over 1000 images of flowers, sampled from 17 different species. The dataset comes from the Visual Geometry Group at the University of Oxford, and full details of the data can be found here.

For this exercise, you should write some code which does the following:

1. Define a particular image that you want to work with
2. For that image
  - Extract the colour histogram using OpenCV
  - Extract colour histograms for all of the *other images in the data
  - Compare the histogram of our chosen image to all of the other histograms
  - For this, use the cv2.compareHist() function with the cv2.HISTCMP_CHISQR metric
  - Find the five images which are most simlar to the target image
  - Save a CSV file to the folder called out, showing the five most similar images and the distance metric:

    |Filename|Distance|
    |---|---|
    |target|0.0|
    |filename1|---|
    |filename2|---|

This assignment is designed to test that you can:

1. Work with larger datasets of images
2. Extract structured information from image data using OpenCV
3. Quantaitively compare images based on these features, performing distant viewing

## Methods

In this assignment, I have used ```OpenCV``` to design a simple image search algorithm. First I have randomly chosen an image I'd like to work with. Then a colour histogram was extracted using ```OpenCV``` for that image first and then for the rest of the images. Then histogramss were compared and the five most similar images were chosen.

## File structure

- In the ```src``` folder you can find the script, in a ```.py``` format.
- The ```archive``` folder contains the older version of the script, in a Jupyter Notebook format.
- The five most similar images aand the distance metrics are saved in the ```out``` folder, in the following format:

|Filename|Distance|
|---|---|
|target|0.0|
|filename1|---|
|filename2|---|

- The data is available in the ```flowers``` folder or via [this link](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/). It is a collection of over 1000 images of flowers, sampled from 17 different species.
- The ```setup.sh``` file includes everything to run the installments.
- The ```run.sh``` file includes everything to run the script.

## How to run it

1. Clone this reepository on your own device
2. Open Terminal at this folder or type ```cd assignment1-simple-image-search-julifurjes-main```
3. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
4. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment

## Notes

Make sure that you have all the files in your cloned folder and that your ```flowers``` folder is not empty.

## Results

The output file called ```top_five.csv``` contains the five most resembling pictures to the randomly chosen flower picture.
