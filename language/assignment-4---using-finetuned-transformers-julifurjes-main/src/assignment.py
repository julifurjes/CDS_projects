from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def emotion_classifying():
    classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      return_all_scores=True) # loading the emotion classification
    data = pd.read_csv('data/fake_or_real_news.csv', index_col=0)
    headlines = data['title'][:1000].astype(str).values.tolist() # taking the top 1000 headlines
    label = data['label'][:1000] # taking the top 1000 labels
    preds = classifier(headlines) # classifying
    # creating an empty list for each emotion
    anger = [0] * len(headlines)
    disgust = [0] * len(headlines)
    fear = [0] * len(headlines)
    joy = [0] * len(headlines)
    neutral = [0] * len(headlines)
    sadness = [0] * len(headlines)
    surprise = [0] * len(headlines)
    for i in range(len(headlines)): # saving the scores
        anger[i] = preds[i][0].get("score")
        disgust[i] = preds[i][1].get("score")
        fear[i] = preds[i][2].get("score")
        joy[i] = preds[i][3].get("score")
        neutral[i] = preds[i][4].get("score")
        sadness[i] = preds[i][5].get("score")
        surprise[i] = preds[i][6].get("score")
    data = pd.DataFrame(list(zip(headlines,label, anger, disgust, fear, joy, neutral, sadness, surprise)), columns=['title', 'label', 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    return preds, headlines, data

def finding_max(data):
    data['top_emotion'] = data[['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']].idxmax(axis = 1)
    print(data)
    return data

def visualisations(data):
    sns.countplot(x=data['top_emotion']) # saving all titles
    plt.savefig('out/all_titles.png') # saving the output
    real_df = data[data['label'] == 'REAL'] # filtering for real titles
    sns.countplot(x=real_df['top_emotion']) # saving real titles
    plt.savefig('out/real_titles.png') # saving the output
    fake_df = data[data['label'] == 'FAKE'] # filtering for fake titles
    sns.countplot(x=fake_df['top_emotion']) # saving fake titles
    plt.savefig('out/fake_titles.png') # saving the output

def main():
    prediction, headline, df = emotion_classifying()
    df = finding_max(df)
    visualisations(df)

if __name__ == '__main__':
    main()