# Extra Assignment

## About the assignment

In this assignment, I have trained a classifier on a dataset containing pictures of Santa and non-Santa. The dataset is not included in the repository, due to its great size, but it can be downloaded from [here](https://www.kaggle.com/datasets/deepcontractor/is-that-santa-image-classification). 

The point of this software is that after training the model, there is an interactive part where the user is asked in a prompt to choose whether the program should test a Santa or a non-Santa picture. Based on the input (either 'S' or 'N'), the program will randomly choose a picture from the given category and test the model on it. Then it prints out its prediction about the picture, which hopefully matches the user's input.

NB: My code also includes the unzipping process, therefore if you have already unzipped it before uploading it, upload the folder with its original name ('is that santa'). This way the unzipping will be automatically skipped.

NB 2: For some reason the user input classification sometimes works perfectly, and sometimes it does not wand I could not figure out why. Therefore if it does not work, a prompt pops up, suggesting that the user should rerun the code.

## File structure

- In the ```src``` folder you can find the script, in a ```.py``` format.
- The training and validation history plots are saved in the ```out``` folder, called ```eval.png```. Besides this plot, a plot representing some examples from both cateegory can be also found under the name ```examples.png```.
- The ```setup.sh``` file includes everything to run the installments.
- The ```run.sh``` file includes everything to run the script.

## How to run it
First download the dataset from the above mentioned link. Then write ```bash setup.sh``` in the terminal to install the requirements and virtual environment, and run ```bash run.sh``` to run the script.