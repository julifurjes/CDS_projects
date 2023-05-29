# Assignment 1: Linguistic analysis using NLP

## About the assignment

In this assignment, I have extracted information from The Uppsala Student English Corpus (USE). All of the data is included in the folder called ```data``` but you can access more documentation via [this link](https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2457).

First, I have created overall statistics of the corpus, then this statistics was also calculated for each text file.
The following information was extracted:
- Relative frequency of Nouns, Verbs, Adjective, and Adverbs per 10,000 words
- Total number of unique PER, LOC, ORGS

Then, the report for each text file was saved in tables, one for each subfolder. The structure of the tables is the following:

|Filename|RelFreq NOUN|RelFreq VERB|RelFreq ADJ|RelFreq ADV|Unique PER|Unique LOC|Unique ORG|
|---|---|---|---|---|---|---|---|
|file1.txt|---|---|---|---|---|---|---|
|file2.txt|---|---|---|---|---|---|---|
|etc|---|---|---|---|---|---|---|

This assignment is designed to test that you can:

1. Work with multiple input data arranged hierarchically in folders;
2. Use ```spaCy``` to extract linguistic information from text data;
3. Save those results in a clear way which can be shared or used for future analysis

## File structure

- In the ```data/USEcorpus``` folder you can find the dataset. It contains 14 subfolders. 
- In the ```out``` folder, you can find the above mentioned tables. Each subfolder has a corresponding ```.csv``` file.
- In the ```src``` folder, you can find the script, both in an ```.ipynb``` and in a ```.py``` format.
- The required packages are listed in the ```requirements.txt``` file, which you can run with ```setup.sh```.
- The ```run.sh``` file includes everything to run the script.

## How to run it

1. Clone this repository on your own device
2. Open Terminal at this folder or type ```cd assignment1-simple-image-search-julifurjes-main```
3. Run ```bash setup.sh``` in the terminal at this folder to install the requirements and virtual environment
4. Run ```bash run.sh``` to run the script itself, in the previously created virtual environment

## Notes

Make sure that you have all the files in your cloned folder and that your ```data/USEcorpus``` folder has all the 14 subfolders.
