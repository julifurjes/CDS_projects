import spacy
nlp = spacy.load("en_core_web_sm")
import os, glob
import nltk
import numpy as np
import re
import pandas as pd

# predefine functions

def calc_rel_freq(doc, word_type):
    count = 0
    for doc_units in doc:
        for token in doc_units:
            if token.pos_ == word_type:
                count += 1
    freq = round((count/len(doc)) * 10000, 2) # calculating frequency
    print('Relative frequency of ', word_type, ':', freq)

def calc_ent(doc, ent_type):
    list = [""] # creating empty list
    for doc_units in doc:
        nlp_doc_units = nlp(doc_units)
        for ent in nlp_doc_units.ents:
            if(ent.label_ == ent_type):
                list.append(ent.text)
    ent_count = len(set(list))
    print("Number of unique PERs:", ent_count)

# running functions

def load_data():
    folder_path = 'data/USEcorpus' # assign directory
    corpus = [""] * 1497 # creating empty arrays for the corpus
    i = 0
    for dirpath, dirnames, files in os.walk(folder_path):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            with open(os.path.join(dirpath, file_name), "r", encoding="latin-1") as f:
                corpus[i] = f.read() # read txt file
        if(i < 1497):
            i = i + 1
    return corpus

def clean_data(corpus):
    i = 0
    for text in corpus:
        corpus[i] = re.sub('<.*?>', '', text) # removing special characters
        corpus[i] = re.sub('[\n\t]', ' ', corpus[i]) # removing enters and tabs
        i = i + 1
    doc = [""] * 1497 # creating empty lists for the tokens
    for i in range(1,1497):
        doc[i] = nlp(corpus[i]) # creating a doc object
    return doc

def rel_freq(doc):
    # nouns
    word_type = "NOUN"
    calc_rel_freq(doc, word_type)
    # verbs
    word_type = "VERB"
    calc_rel_freq(doc, word_type)
    # adjectives
    word_type = "ADJ"
    calc_rel_freq(doc, word_type)
    # adverbs
    word_type = "ADV"
    calc_rel_freq(doc, word_type)

def creating_entities(doc):
    # create empty lists
    entities_words = []
    entities_labels = []
    entities = []
    # add each entity to list
    for doc_units in doc:
        nlp_doc_units = nlp(doc_units)
        for ent in nlp_doc_units.ents:
            entities_words.append(ent.text)
            entities_labels.append(ent.label_)
            entities.append(ent.text)
    print(set(entities_words))
    print(set(entities_labels))

def total_numbers(doc):
    # PER
    ent_type = "PER"
    calc_ent(doc, ent_type)
    # LOC
    ent_type = "LOC"
    calc_ent(doc, ent_type)
    # ORG
    ent_type = "ORG"
    calc_ent(doc, ent_type)

def save_table():
    i = 0
    noun_count = verb_count = adj_count = adv_count = 0
    per_list = loc_list = org_list = [""]
    folder_path = 'data/USEcorpus' # assign directory
    for dirpath, dirnames, files in os.walk(folder_path):
        table = pd.DataFrame(columns=['Filename', 'RelFreq NOUN', 'RelFreq VERB', 'RelFreq ADJ', 'RelFreq ADV', 'Unique PER', 'Unique LOC', 'Unique ORG'])
        for file_name in files:
            with open(os.path.join(dirpath, file_name), "r", encoding="latin-1") as f:
                text = f.read()
                text = nlp(text)
                for token in text:
                    if token.pos_ == "NOUN":
                        noun_count += 1
                    if token.pos_ == "VERB":
                        verb_count += 1
                    if token.pos_ == "ADJ":
                        adj_count += 1
                    if token.pos_ == "ADV":
                        adv_count += 1
                noun_freq = round((noun_count/len(text)) * 10000, 2)
                verb_freq = round((verb_count/len(text)) * 10000, 2)
                adj_freq = round((adj_count/len(text)) * 10000, 2)
                adv_freq = round((adv_count/len(text)) * 10000, 2)
                for ent in text.ents:
                    if(ent.label_ == 'PERSON'):
                        per_list.append(ent.text)
                    if(ent.label_ == 'LOC'):
                        loc_list.append(ent.text)
                    if(ent.label_ == 'ORG'):
                        org_list.append(ent.text)
                table.loc[i] = [file_name, noun_freq, verb_freq, adj_freq, adv_freq, len(set(per_list)), len(set(loc_list)), len(set(org_list))]
            i = i + 1
            noun_count = verb_count = adj_count = adv_count = 0
            per_list = loc_list = org_list = [""]
        table.to_csv('out/' + dirpath[-2:] + '.csv', index=False)
        i = 0

def main():
    corpus = load_data()
    doc = clean_data(corpus)
    rel_freq(doc)
    creating_entities(doc)
    total_numbers(doc)
    save_table()

if __name__ == '__main__':
    main()