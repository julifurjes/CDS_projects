# Assignment 4: Using finetuned transformers

## About the assignment

In this assignment, I have extracted information from the Fake or Real News dataset and performed emotion classification.
NB: The dataset is not uploaded due to its size, but can be downloaded from [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news).For this purpose, a HuggingFace model was used, called ```j-hartmann/emotion-english-distilroberta-base```, what you can read more about [here](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base).

Furthermore, only 1000 datapoints were used in the run, due to time efficiency issues. However, this can be easily changed by removing the indexing when defining the headlines and labels.

## File structure

- In the ```out``` folder, you can find some visualisations for:
  - Emotion distribution across all headlines.
  - Emotion distribution across real headlines.
  - Emotion distribution across fake headlines.
- In the ```src``` folder, you can find the script.
- The required packages are listed in the ```requirements.txt``` file, which you can run with ```setup.sh```.
- The ```run.sh``` file includes everything to run the script.

## How to run it

1. Clone this repository on your own device
2. Open Terminal at this folder or type ```cd assignment-4---using-finetuned-transformers-julifurjes-main```
3. Make sure that you have the data downloaded from [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) and located in a folder called ```data```
4. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
5. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment

## Results

You can find the diagrams generated from the script in the ``out``` folder. However, here are some of my observations:

- While with all titles, the neutral is way the most popular category, with real and fake titles, it's digsust. Which doesn't make sense??
- Fake titles have a great amount of sadness, while real titles have a great amount of joy. This can indicate that fake titles are more negative.
