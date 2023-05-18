# Assignment 1: Building a simple image search algorithm

## About the assignment

In this assignment, I have used ```OpenCV``` to design a simple image search algorithm. First I have randomly chosen an image I'd like to work with. Then a colour histogram was extracted using ```OpenCV``` for that imgae first and then for the rest of the images. Then histogramss were compared and the five most similar images were chosen.

## File structure

- In the ```src``` folder you can find the script, in a ```.py``` format.
- The ```archive``` folder contains the older version of the script, in a Jupyter Notebook format.
- The five most similar images aand the distance metrics are saved in the ```out``` folder, in the following format:

|Filename|Distance]
|---|---|
|target|0.0|
|filename1|---|
|filename2|---|

- The data is available in the ```flowers``` folder or via [this link](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/). It is a collection of over 1000 images of flowers, sampled from 17 different species.
- The ```setup.sh``` file includes everything to run the installments.
- The ```run.sh``` file includes everything to run the script.

## How to run it
First write ```bash setup.sh``` in the terminal to install the requirements and virtual environment, and run ```bash run.sh``` to run the script.
