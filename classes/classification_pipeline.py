from classes.dense_transformer import DenseTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score


class ClassificationPipeline():
    def __init__(self, clf_id, clf, vectorizer, feature_processing, pipe=None):
        self.pipe = pipe 
        self.clf_id = clf_id 
        self.clf = clf
        self.vectorizer = vectorizer
        self.feature_processing = feature_processing
                   
    def create_feature_pipeline(self, n_components, memory):
        self.pipe = Pipeline([
            ('feature_pipeline', FeatureUnion([
                ('text', Pipeline([
                    ('vectorizer', self.vectorizer),
                    ('to_dense', DenseTransformer()),
                    ('lda', LinearDiscriminantAnalysis(n_components=n_components)),
                ])),
                ('feature_processing', self.feature_processing)
            ])),
            (self.clf_id, self.clf)
        ], memory=memory)
    
    def create_pipeline(self, n_components, memory):
        self.pipe = Pipeline([
            ('feature_pipeline', FeatureUnion([
                ('text', Pipeline([
                    ('vectorizer', self.vectorizer),
                    ('to_dense', DenseTransformer()),
                    ('lda', LinearDiscriminantAnalysis(n_components=n_components)),
                ])),
            ])),
            (self.clf_id, self.clf)
        ], memory=memory)
            
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, memory, n_components=4, use_features=True):
        if use_features:
            self.create_feature_pipeline(n_components=n_components, memory=memory)
        else:
            self.create_pipeline(n_components=n_components, memory=memory)
            
        self.pipe.fit(X_train, y_train)
        preds = self.pipe.predict(X_val)
        
        accuracy = accuracy_score(y_val, preds)
        precision = precision_score(y_val, preds, average='macro')
        recall = recall_score(y_val, preds, average='macro')
        f1 = f1_score(y_val, preds, average='macro')
        kappa = cohen_kappa_score(y_val, preds)
        
        return accuracy, precision, recall, f1, kappa
    