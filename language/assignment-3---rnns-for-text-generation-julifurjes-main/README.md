# Assignment 3: Rnns for text generation

## About the assignment

In this assignment, I have trained a text generation model on comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

First I have trained a model on the Comments section of the data, then the trained model was saved, so it would not have to be trained every time it should be used. Then this saved model was loaded form the ```my_model``` folder, so based on this it could generate a sentence based on a user-suggested word in the prompt field.

The code consists of three scripts:
1. ```assignment.py``` - this is where the model gets created and saved
2. ```loading_model.py``` - this is where the model gets loaded and the text generated from the user prompt
3. ```predefined_functions.py``` - this is where I store the functions that I do not define in the other scripts

This assignment is designed to test that you can:

- Use TensorFlow to build complex deep learning models for NLP
- Illustrate that you can structure repositories appropriately
- Provide clear, easy-to-use documentation for your work.

NB: I only used 3000 datapoints and 5 epochs due to time efficiency and computer capacity. But this caan be easily changed by altering the epoch size and removing the sampling line.

## File structure

- The ```my_model``` folder contains the model saved from the training process.
- In the ```src``` folder, you can find the above mentioned three scripts.
- The required packages are listed in the ```requirements.txt``` file, which you can run with ```setup.sh```.
- The ```run.sh``` file includes everything to run the script.

## How to run it

1. Clone this repository on your own device
2. Open Terminal at this folder or type ```cd assignment-3---rnns-for-text-generation-julifurjes```
3. Make sure you have the data downloaded from [here](https://www.kaggle.com/datasets/aashita/nyt-comments) and saved it in a folder called ```data```
4. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
5. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment

## Results

Since I only used limited amount of data, the output of the sentence generation is not great, however, that could be easily solved by changing the data and the epoch size as mentioned above.
For example, the input of the word 'cat' creates the following sentence:
'Cat the same and the the own the the own the'
