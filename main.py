#!/usr/bin/env python3
"""
Modified liberally from https://dagshub.com/docs/experiment-tutorial/2-data-versioning/
The goal of this repository is not to learn how to create an NLP classifier but to learn how to use DVC, therefore the ML code is unimportant.
The training process has been seperated into distinct stages in order to demonstrate DVC's pipeline and experiment-tracking features.
"""

import sys
import yaml
import pickle
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score

_params = None
def read_param(name, filename="params.yaml"):
    global _params
    if _params == None:
        with open(filename) as file:
            _params = yaml.safe_load(file)
    obj = _params
    while "." in name:
        key, name = name.split(".", 1)
        obj = obj[key]
    return obj[name]
    

def split(dataset_path=read_param("paths.dataset"),
          train_df_path=read_param("paths.train_df"),
          test_df_path=read_param("paths.test_df"),
          random_state=read_param("split.seed")):
    
    print(f"Loading {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['MachineLearning'] = df['Tags'].str.contains('machine-learning').fillna(False)
    train_df, test_df = train_test_split(df, random_state=random_state, stratify=df['MachineLearning'])
    
    print(f"Saving {train_df_path}, {test_df_path}")
    train_df.to_csv(train_df_path)
    test_df.to_csv(test_df_path)

def featurize(train_df_path=read_param("paths.train_df"),
              test_df_path=read_param("paths.test_df"),
              train_df_featurized_path=read_param("paths.train_df_featurized"),
              test_df_featurized_path=read_param("paths.test_df_featurized")):
    
    def feature_engineering(df):
        """Stolen directly from DAGsHub tutorial"""
        df['CreationDate'] = pd.to_datetime(df['CreationDate'])
        df['CreationDate_Epoch'] = df['CreationDate'].astype('int64') // 10 ** 9
        df = df.drop(columns=['Id', 'Tags'])
        df['Title_Len'] = df.Title.str.len()
        df['Body_Len'] = df.Body.str.len()
        # Drop the correlated features
        df = df.drop(columns=['FavoriteCount'])
        df['Text'] = df['Title'].fillna('') + ' ' + df['Body'].fillna('')
        return df
    
    print(f"Featurizing {train_df_path} to {train_df_featurized_path}")
    feature_engineering(pd.read_csv(train_df_path)).to_csv(train_df_featurized_path)
    
    print(f"Featurizing {test_df_path} to {test_df_featurized_path}")
    feature_engineering(pd.read_csv(test_df_path)).to_csv(test_df_featurized_path)

def tfidf(train_df_featurized_path=read_param("paths.train_df_featurized"),
          test_df_featurized_path=read_param("paths.test_df_featurized"),
          train_tfidf_path=read_param("paths.train_tfidf"),
          test_tfidf_path=read_param("paths.test_tfidf"),
          tfidf_path=read_param("paths.tfidf"),
          max_features=read_param("tfidf.max_features")):
    
    print(f"Loading {train_df_featurized_path} and {test_df_featurized_path}")
    train_df = pd.read_csv(train_df_featurized_path)
    test_df = pd.read_csv(test_df_featurized_path)
    
    print(f"Training TF-IDF vectorizer with max_features={max_features}")
    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf.fit(train_df['Text'])
    
    print(f"Transforming {train_df_featurized_path} to {train_tfidf_path}")
    train_tfidf = tfidf.transform(train_df['Text'])
    sp.save_npz(train_tfidf_path, train_tfidf)
    
    print(f"Transforming {test_df_featurized_path} to {test_tfidf_path}")
    test_tfidf = tfidf.transform(test_df['Text'])
    sp.save_npz(test_tfidf_path, test_tfidf)
    
    print(f"Writing tfidf vectorizer to {tfidf_path}")
    pickle.dump(tfidf, open(tfidf_path, 'wb'))

def _eval_model(model, X, y):
    """Stolen directly from DAGsHub tutorial"""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return {
        'roc_auc': roc_auc_score(y, y_proba),
        'average_precision': average_precision_score(y, y_proba),
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }

def train(train_df_featurized_path=read_param("paths.train_df_featurized"),
          train_tfidf_path=read_param("paths.train_tfidf"),
          loss=read_param("train.loss"),
          random_state=read_param("train.seed"),
          model_path=read_param("paths.model")):
    
    print(f"Loading {train_df_featurized_path} and {train_tfidf_path}")
    train_df = pd.read_csv(train_df_featurized_path)
    train_tfidf = sp.load_npz(train_tfidf_path)
    
    print(f"Training SGDClassifier model with loss={loss}, random_state={random_state}")
    model = SGDClassifier(loss=loss, random_state=random_state)
    model.fit(train_tfidf, train_df['MachineLearning'])
    
    print(f"Writing model to {model_path}")
    pickle.dump(model, open(model_path, 'wb'))
    
    print("Evaluating model on training set:")
    print(_eval_model(model, train_tfidf, train_df['MachineLearning']))

def test(test_df_featurized_path=read_param("paths.test_df_featurized"),
         test_tfidf_path=read_param("paths.test_tfidf"),
         model_path=read_param("paths.model")):
    
    print(f"Loading {test_df_featurized_path} and {test_tfidf_path}")
    test_df = pd.read_csv(test_df_featurized_path)
    test_tfidf = sp.load_npz(test_tfidf_path)
    
    print(f"Loading model from {model_path}")
    model = pickle.load(open(model_path, 'rb'))
    
    print("Evaluating model on test set:")
    print(_eval_model(model, test_tfidf, test_df['MachineLearning']))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        sys.exit(f"Usage: python3 {sys.argv[0]} [split|featurize|tfidf|train|test]")
    elif sys.argv[1] == "split":
        split()
    elif sys.argv[1] == "featurize":
        featurize()
    elif sys.argv[1] == "tfidf":
        tfidf()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
    else:
        sys.exit(f"Invalid operation {sys.argv[1]}\nUsage: python3 {sys.argv[0]} [split|tfidf|train|test]")
