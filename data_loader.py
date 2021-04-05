import os
import json
import pandas as pd 
import numpy as np

from preprocessing import Preprocessor
from sklearn.utils import shuffle
from json import JSONEncoder
from collections import defaultdict 
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
    
    
class PhraseCount:
    def __init__(self):
        self.label = ""
        self.count = 0
        
class Encoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

    
class DataLoader:
    def __init__(self):
        self.phrases = pd.read_csv(DATA_PATH + '/dictionary.txt', delimiter = "|", names=['phrase', 'phrase_id1'])
        self.labels = pd.read_csv(DATA_PATH + '/sentiment_labels.txt', delimiter = '|')
        self.labels.columns = ['phrase_id2', 'sentiment_val']
        
    def create_dataframe(self, preprocess=True, split=True, remove_duplicates=False): 
        df = self.phrases.set_index('phrase_id1').join(self.labels.set_index('phrase_id2'),
                                                                       how='inner').rename_axis(index='phrase_id').reset_index()
        df['label_id'] = df.apply(lambda row: label_sentiment_id(row), axis=1)
        df['label'] = df.apply(lambda row: label_sentiment_name(row), axis=1)
        df = df.dropna()
            
        if preprocess:
            print('Preprocessing...')
            phrases = df['phrase'].values.tolist()
            phrases_cleaned = [Preprocessor(phrase).preprocess() for phrase in tqdm(phrases)]
            df['phrase_clean'] = phrases_cleaned
            
            filter = df['phrase_clean'] != ""
            df = df[filter]
            df = df[['phrase_id', 'phrase', 'phrase_clean', 'sentiment_val', 'label_id', 'label']]
              
        if split:
            return self.split(df, remove_duplicates=remove_duplicates)
        else:
            return df 
        
    def dedup(self, df):
        print('Deduplicating...')
        
        temp = df
        temp['word_count'] = temp['phrase_clean'].apply(lambda x: len(x.split()))
        temp = temp[temp['word_count'] > 4]
        
        phrases = temp['phrase_clean']
        dups = temp[phrases.isin(phrases[phrases.duplicated()])]
        
        df = pd.concat([df, dups]).drop_duplicates(keep=False)
        
        return df 
        
    def split(self, df, remove_duplicates=False):
        train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
        
        if remove_duplicates:
            train = self.dedup(df=train) 
        
        return shuffle(train), validate, test
    