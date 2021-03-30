import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = stopwords.words('english')

class Preprocessor():
    def __init__(self, text):
        self.text = text
        self.vocab = set()
        self.og_text = text
        self.ps = PorterStemmer()
       
    def lowercase(self):
        self.text = self.text.lower()
    
    def remove_punct(self):         
        self.text = re.sub("[^a-zA-Z.!? ]+", '', self.text) # Remove punctuation except '!' and '." 
        self.text = re.sub(' +', ' ', self.text)
    
    def remove_stopwords(self, stop_words):
        tokens = self.text.split()
        self.text = [word.rstrip()
                     for word in tokens if word not in stop_words]
        self.text = ' '.join(self.text)
    
    def stem(self):
        self.text = self.text.split()
        self.text = [self.ps.stem(word) for word in self.text]
        self.text = ' '.join(self.text)
    
    def tokenize(self):
        self.text = self.text.split()
            
    def join_tokens(self):
        self.text = ' '.join(self.text)
    
    def return_text(self):
        return self.text
    
    def preprocess(self):
        #self.lowercase()
        self.remove_punct()
        self.remove_stopwords(stop_words)
        self.stem()
        self.tokenize()
        self.join_tokens()
        return self.return_text()