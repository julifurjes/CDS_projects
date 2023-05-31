# Language Analytics Project

## About the assignment

In this assignment, I have trained a neural network classifier to predict personality types. I have decided to use the MBTI personality types as a dataset. The Myers-Briggs Type Indicator (MBTI) is a widely used personality assessment tool that is based on the theories of Swiss psychiatrist Carl Jung. It aims to classify individuals into different personality types, providing insights into their preferences, behaviors, and tendencies.

The MBTI is grounded in Jung's theory of psychological types, which suggests that individuals possess innate preferences for how they perceive and interact with the world. According to Jung, there are four dichotomies that shape personality:
a. Extraversion (E) - Introversion (I): Refers to the preferred focus of attention, either outward or inward.
b. Sensing (S) - Intuition (N): Describes how individuals gather information, either through the five senses or through patterns and possibilities.
c. Thinking (T) - Feeling (F): Reflects how individuals make decisions, either through logic and analysis or through values and emotions.
d. Judging (J) - Perceiving (P): Captures how individuals approach the external world, either through organization and structure or through flexibility and spontaneity.

Combining the preferences from each dichotomy results in a four-letter type. There are sixteen possible combinations, such as ISTJ, ENFP, or INFJ, each representing a unique personality type.

If you want to fill out the test yourself and see how it works, you can do so [here](https://www.16personalities.com/free-personality-test).

The dataset consists of over 8600 rows of data, on each row is a person’s:

- Type (This persons 4 letter MBTI code/type)
- A section of each of the last 50 things they have posted (Each entry separated by "|||" (3 pipe characters))

The data can be downloaded from [here](https://www.kaggle.com/datasets/datasnaek/mbti-type).

After training the neural network classifier, I have saved the models and then loaded them shortly afterwards again. This way the training can be easily disabled for time efficiency. After loading them, the user is asked to write a sentence, so the classifier can predict which personality type does it belong to.

NB: I did not separate the posts, just removed the special characters, including the separating ones. I have decided to do this because since they still belong to the same label, it does not really influence the results if I don't separate them.

## File structure

- The ```out``` folder contains:
    - ```classifier.joblib```
    - ```neural_network_classifier.txt```
    - ```tfidf_vectorizer.joblib```
- In the ```src``` folder, you can find the script.
- The required packages are listed in the ```requirements.txt``` file, which you can run with ```setup.sh```.
- The ```run.sh``` file includes everything to run the script.

## How to run it

1. Clone this repository on your own device
2. Open Terminal at this folder or type ```cd assignment-4---using-finetuned-transformers-julifurjes-main```
3. Make sure that you have the data downloaded from [here](hhttps://www.kaggle.com/datasets/datasnaek/mbti-type) and located in a folder called ```data```
4. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
5. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment

## Results

The prediction is not completely functional, maybe due to the small data size.