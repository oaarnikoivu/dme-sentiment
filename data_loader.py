import os
import pandas as pd 

from preprocessing import Preprocessor
from sklearn.utils import shuffle
from tqdm import tqdm


DATA_PATH = os.path.abspath('stanfordSentimentTreebank/')


def label_sentiment_id(row):
    if row['sentiment_val'] > 0 and row['sentiment_val'] <= 0.2:
        return 1 # Very negative
    if row['sentiment_val'] > 0.2 and row['sentiment_val'] <= 0.4:
        return 2 # negative
    if row['sentiment_val'] > 0.4 and row['sentiment_val'] <= 0.6:
        return 3 # Neutral
    if row['sentiment_val'] > 0.6 and row['sentiment_val'] <= 0.8:
        return 4 # positive
    if row['sentiment_val'] > 0.8 and row['sentiment_val'] <= 1:
        return 5 # Very positive

def label_sentiment_name(row):
    if row['label_id'] == 1:
        return 'Very negative'
    if row['label_id'] == 2:
        return 'Negative'
    if row['label_id'] == 3:
        return 'Neutral'
    if row['label_id'] == 4:
        return 'Positive'
    if row['label_id'] == 5:
        return 'Very positive'

    
class DataLoader:
    def __init__(self):
        self.phrases = pd.read_csv(DATA_PATH + '/dictionary.txt', delimiter = "|", names=['phrase', 'phrase_id1'])
        self.labels = pd.read_csv(DATA_PATH + '/sentiment_labels.txt', delimiter = '|')
        self.labels.columns = ['phrase_id2', 'sentiment_val']
       
    def create_dataframe(self, remove_duplicates=True, preprocess=True): 
        df = self.phrases.set_index('phrase_id1').join(self.labels.set_index('phrase_id2'),
                                                                       how='inner').rename_axis(index='phrase_id').reset_index()
        df['label_id'] = df.apply(lambda row: label_sentiment_id(row), axis=1)
        df['label'] = df.apply(lambda row: label_sentiment_name(row), axis=1)
        df = df.dropna()
            
        if preprocess:
            phrases = df['phrase'].values.tolist()
            phrases_cleaned = [Preprocessor(phrase).preprocess() for phrase in tqdm(phrases)]
            df['phrase_clean'] = phrases_cleaned
            
        if remove_duplicates:
            df = df.drop_duplicates(subset=['phrase_clean']) 
                 
        filter = df['phrase_clean'] != ""
        df = df[filter]
                
        df = shuffle(df)
        df = df[['phrase_id', 'phrase', 'phrase_clean', 'sentiment_val', 'label_id', 'label']]

        return df
